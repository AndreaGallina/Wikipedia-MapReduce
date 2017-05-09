package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.random.XORShiftRandom;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.io.Serializable;

public class KMedoids implements Serializable{
	private double epsilon = 1e-4;
	private int maxIterations;
	private int k;
	private long seed;

	private double eps = 1.0;

	public KMedoids(int k, int maxIterations, long seed){
		this.k = k;
		this.maxIterations = maxIterations;
		// this.seed = new Random().nextLong();
		this.seed = seed;

		while ((1.0 + (eps / 2.0)) != 1.0) {
	      eps /= 2.0;
	    }
	}

	public void run(JavaRDD<Vector> data, JavaSparkContext sc)
	{
		JavaRDD<Double> norms = data.map((v) -> Vectors.norm(v, 2.0));
		norms.cache();
		JavaRDD<VectorWithNorm> zippedData = data.zip(norms).map((tuple) -> new VectorWithNorm(tuple._1,tuple._2));
		
		runAlgorithm(zippedData,sc);
	}

	private void runAlgorithm(JavaRDD<VectorWithNorm> data, JavaSparkContext sc){

		List<VectorWithNorm> centers = new ArrayList<>(initRandom(data));

		boolean converged = false;
		double cost = 0.0;
		int iteration = 0;

		while(iteration < maxIterations && !converged)
		{
			DoubleAccumulator costAccum = sc.sc().doubleAccumulator();
			Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);

			Map<Integer,Tuple2<Vector,Long>> totalContribs = data.mapPartitionsToPair((points) -> {
				List<VectorWithNorm> thisCenters = bcCenters.value();
				int dims = thisCenters.get(0).vector.size();

				List<Vector> sums = new ArrayList<Vector>();
				long[] counts = new long[thisCenters.size()];
				for(int i=0; i<thisCenters.size(); i++)
					sums.add(Vectors.zeros(dims));
				Arrays.fill(counts, 0L);

				while(points.hasNext()){
					VectorWithNorm point = points.next();
					Tuple2<Integer, Double> bestPartition = findClosest(thisCenters, point);
					int bestCenter = bestPartition._1;
					double pointCost = bestPartition._2;
					costAccum.add(pointCost);
					Vector sum = sums.get(bestCenter);
					BLAS.axpy(1.0, point.vector, sum);
					counts[bestCenter]++;
				}

				List<Tuple2<Integer, Tuple2<Vector, Long>>> partitionCounts = new ArrayList<>();

				for(int i=0; i < counts.length; i++)
				{
					if(counts[i]>0)
						partitionCounts.add(new Tuple2<>(i, new Tuple2<>(sums.get(i), counts[i])));
				}

				return partitionCounts.iterator();
			}).reduceByKey((tuple1, tuple2) -> {

				Vector sum1 = tuple1._1;
				Vector sum2 = tuple2._1;
				long count1 = tuple1._2;
				long count2 = tuple2._2;

				BLAS.axpy(1.0, sum2, sum1);

				return new Tuple2<>(sum1, count1 + count2);
			}).collectAsMap();

			bcCenters.destroy();

			converged = true;
			for(Map.Entry<Integer, Tuple2<Vector,Long>> entry : totalContribs.entrySet())
			{
				int i = entry.getKey();
				Vector sum = entry.getValue()._1;
				double count = entry.getValue()._2;
				
				BLAS.scal(1.0 / count, sum);
				VectorWithNorm newCenter = new VectorWithNorm(sum);
				if (converged && fastSquaredDistance(newCenter, centers.get(i)) > epsilon*epsilon)
					converged = false;
				centers.set(i, newCenter);
			}
			cost = costAccum.value();
			iteration++;
		}
		System.out.println("KMeans converged in "+iteration+" iterations with cost "+cost);
	}

	private List<VectorWithNorm> initRandom(JavaRDD<VectorWithNorm> data)
	{
		return data.takeSample(false, k, new XORShiftRandom(seed).nextInt());
			
	}

	private Tuple2<Integer,Double> findClosest(List<VectorWithNorm> centers, VectorWithNorm point)
	{
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		int i = 0;
		for(VectorWithNorm center : centers)
		{
			double lowerBoundOfSqDistance = center.norm - point.norm;
			lowerBoundOfSqDistance = lowerBoundOfSqDistance * lowerBoundOfSqDistance;
			if(lowerBoundOfSqDistance < bestDistance)
			{
				double distance = fastSquaredDistance(center, point);
				if(distance < bestDistance)
				{
					bestDistance = distance;
					bestIndex = i;
				}
			}
			i++;
		}
		return new Tuple2<>(bestIndex, bestDistance);
	}


	/**
	 * Method derived from the class org.apache.spark.mllib.util.MLUtils and reimplemented
	 * in Java since the spark version is private.
	 * @param  v1 [first vector]
	 * @param  v2 [second vector]
	 * @return    [squared distance between vectors]
	 */
	private double fastSquaredDistance(VectorWithNorm v1, VectorWithNorm v2){
		Vector vec1 = v1.vector;
		Vector vec2 = v2.vector;
		double norm1 = v1.norm;
		double norm2 = v2.norm;
		double precision = 1e-6;

		double sumSquaredNorm = norm1 * norm1 + norm2 * norm2;
	    double normDiff = norm1 - norm2;
	    double sqDist = 0.0;

	    double precisionBound1 = 2.0 * eps * sumSquaredNorm / (normDiff * normDiff + eps);
	    if(precisionBound1<precision)
	    {
	    	sqDist = sumSquaredNorm - 2.0 * BLAS.dot(vec1,vec2);
	    }
	    else if(vec1 instanceof SparseVector || vec2 instanceof SparseVector)
	    {
	    	double dotValue = BLAS.dot(vec1, vec2);
	    	sqDist = Math.max(sumSquaredNorm - 2.0 * dotValue, 0.0);
	    	double precisionBound2 = eps * (sumSquaredNorm + 2.0 * Math.abs(dotValue)) / (sqDist + eps);
			if (precisionBound2 > precision) {
				sqDist = Vectors.sqdist(vec1, vec2);
			}
	    }
	    else 
	    	sqDist = Vectors.sqdist(v1.vector, v2.vector);

	   	return sqDist;
	}

}
