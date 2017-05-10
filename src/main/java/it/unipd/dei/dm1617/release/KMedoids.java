package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.random.XORShiftRandom;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;
import scala.Tuple3;

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

	public KMedoids(int k, int maxIterations, long seed){
		this.k = k;
		this.maxIterations = maxIterations;
		// this.seed = new Random().nextLong();
		this.seed = seed;
	}

	public void run(JavaRDD<Vector> data, JavaSparkContext sc)
	{
		JavaRDD<Double> norms = data.map((v) -> Vectors.norm(v, 2.0));
		norms.cache();
		JavaRDD<VectorWithNorm> zippedData = data.zip(norms).map((tuple) -> new VectorWithNorm(tuple._1,tuple._2));
		
		runAlgorithmMedoids(zippedData,sc);
	}

	// private void runAlgorithm(JavaRDD<VectorWithNorm> data, JavaSparkContext sc){

	// 	List<VectorWithNorm> centers = new ArrayList<>(initRandom(data));

	// 	boolean converged = false;
	// 	double cost = 0.0;
	// 	int iteration = 0;

	// 	while(iteration < maxIterations && !converged)
	// 	{
	// 		DoubleAccumulator costAccum = sc.sc().doubleAccumulator();
	// 		Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);

	// 		Map<Integer,Tuple2<Vector,Long>> totalContribs = data.mapPartitionsToPair((points) -> {
	// 			List<VectorWithNorm> thisCenters = bcCenters.value();
	// 			int dims = thisCenters.get(0).vector.size();

	// 			// inizializza (tutti i componenti di tutti i vettori di) sums a zero,
	// 			// e tutti i componenti di counts a zero. Sums e counts hanno dimensione
	// 			// pari al numero di centri
	// 			List<Vector> sums = new ArrayList<Vector>();
	// 			long[] counts = new long[thisCenters.size()];
	// 			for(int i=0; i<thisCenters.size(); i++)
	// 				sums.add(Vectors.zeros(dims));
	// 			Arrays.fill(counts, 0L);

	// 			while(points.hasNext()){
	// 				VectorWithNorm point = points.next();
	// 				// ritorna indice del centro più vicino a point e la sua distanza da esso
	// 				Tuple2<Integer, Double> bestPartition = findClosest(thisCenters, point);
	// 				int bestCenter = bestPartition._1;
	// 				double pointCost = bestPartition._2;
	// 				// somma la distanza tra punto e centro più vicino alla costo della funzione obiettivo
	// 				costAccum.add(pointCost);
	// 				// prende il vettore centro di indice corrispondente a bestCenter
	// 				Vector sum = sums.get(bestCenter);
	// 				// somma point a sum (sum = sum + point) (componentwise)
	// 				// la componente i-esima di sum contiene la somma delle componenti i-esime dei vettori assegnati a quel centro		
	// 				BLAS.axpy(1.0, point.vector, sum);
	// 				counts[bestCenter]++;
	// 			}

	// 			List<Tuple2<Integer, Tuple2<Vector, Long>>> partitionCounts = new ArrayList<>();

	// 			// per ogni centro
	// 			for(int i=0; i < counts.length; i++)
	// 			{	
	// 				// se quel centro ha almeno un punto assegnato ad esso, aggiunge a partitionCounts
	// 				// una cosa del tipo <indice del centro, <costo del cluster corrispondente, numero di punti del cluster>>
	// 				if(counts[i]>0)
	// 					partitionCounts.add(new Tuple2<>(i, new Tuple2<>(sums.get(i), counts[i])));
	// 			}

	// 			return partitionCounts.iterator();
	// 		}).reduceByKey((tuple1, tuple2) -> {

	// 			Vector sum1 = tuple1._1;
	// 			Vector sum2 = tuple2._1;
	// 			long count1 = tuple1._2;
	// 			long count2 = tuple2._2;
				
	// 			// somma sum2 a sum1 (sum1 = sum1 + sum2) (componentwise)
	// 			// sta effettuando la somma dei vettori dei punti di un cluster				
	// 			BLAS.axpy(1.0, sum2, sum1);

	// 			return new Tuple2<>(sum1, count1 + count2);
	// 		}).collectAsMap();

	// 		bcCenters.destroy();

	// 		converged = true;
	// 		for(Map.Entry<Integer, Tuple2<Vector,Long>> entry : totalContribs.entrySet())
	// 		{
	// 			int i = entry.getKey();
	// 			Vector sum = entry.getValue()._1;
	// 			double count = entry.getValue()._2;
				
	// 			//sum = (1.0/count) * sum = sum/count (componentwise)
	// 			BLAS.scal(1.0 / count, sum);
	// 			//nuovo centroide
	// 			VectorWithNorm newCenter = new VectorWithNorm(sum);
	// 			if (converged && fastSquaredDistance(newCenter, centers.get(i)) > epsilon*epsilon)
	// 				converged = false;
	// 			centers.set(i, newCenter);
	// 		}
	// 		cost = costAccum.value();
	// 		iteration++;
	// 	}
	// 	System.out.println("KMeans converged in "+iteration+" iterations with cost "+cost);
	// }

	private void runAlgorithmMedoids(JavaRDD<VectorWithNorm> data, JavaSparkContext sc){

		List<VectorWithNorm> centers = new ArrayList<>(initRandom(data));

		boolean converged = false;
		double cost = 0.0;
		int iteration = 0;

		while(iteration < maxIterations && !converged)
		{
			DoubleAccumulator costAccum = sc.sc().doubleAccumulator();
			Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);

			Map<Integer, Tuple2<List<VectorWithNorm>, Double>> totalContribs = data.mapPartitionsToPair((points) -> {
				List<VectorWithNorm> thisCenters = bcCenters.value();
				int dims = thisCenters.get(0).vector.size(); // dimension of a vector representing a document (equal to "vocabSize")

				List<List<VectorWithNorm>> thisPartition = new ArrayList<List<VectorWithNorm>>();
				double[] thisCost = new double[k];
				Arrays.fill(thisCost, 0.0);

				while(points.hasNext()){
					VectorWithNorm point = points.next();
					Tuple2<Integer, Double> bestPartition = findClosestCosine(thisCenters, point);
					// bestPartition._1 contains the index of the cluster to which this point as been assigned
					// bestPartition._2 indicates the cost (i.e. the distance) between the center of the cluster and this point 
					costAccum.add(bestPartition._2);
					thisPartition.get(bestPartition._1).add(point);
					thisCost[bestPartition._1] += bestPartition._2;
				}

				List<Tuple2<Integer, Tuple2<List<VectorWithNorm>, Double>>> partitions = new ArrayList<>();

				// for each cluster, initialize a tuple2 containing the index of the cluster,
				// the list of the points belonging to that cluster, and the total cost of the points for that cluster
				for(int i=0; i<k; i++)
				{
					if(thisPartition.get(i).size()>0)
						partitions.add(new Tuple2<>(i, new Tuple2<>(thisPartition.get(i), thisCost[i])));
				}

				return partitions.iterator();
			}).reduceByKey((tuple1, tuple2) -> {
				// merges all the points belonging to one cluster assigned by the different workers
				List<VectorWithNorm> clusterPoints = new ArrayList<VectorWithNorm>();
				clusterPoints.addAll(tuple1._1);
				clusterPoints.addAll(tuple2._1);

				double clusterCost = tuple1._2 + tuple2._2;
				return new Tuple2<>(clusterPoints, clusterCost);
			}).collectAsMap();

			bcCenters.destroy();

			// Adesso totalContribs contiene tutti i punti appartenenti ad un dato cluster assieme al costo di quel cluster
			// all'interno di una mappa. La struttura è qualcosa di questo tipo:
			// ---------------------------------------------------
			// | ClusterIndex | Points          		  | Cost |
			// ---------------------------------------------------
			// |            0 | ArrayList<VectorWithNorm> | 15.3 |
			// |            1 | ArrayList<VectorWithNorm> | 152.8|
			// ---------------------------------------------------
			// 
			// Quindi per andare a calcolare il nuovo medoide servirà fare qualcosa di questo tipo;
			// 
			// 
			// 
			// Foreach(entry in totalContribs)
			//   int i = entry.getKey()
			//   List<VectorWithNorm> points = entry.getValue()._1
			//   double clusterCost = entry.getValue()._2
			//   double minCost = Double.POSITIVE_INFINITY;
			//   if(clusterCost!=0.0) // Calcola solo se ci sono punti nel cluster
			//   	foreach(point in points) // per ogni punto del cluster i
			//   		double newCost = calculateClusterCost(points, point) // calcola la funzione costo utilizzando come "point" come medoide
			//   		if(newCost<minCost)
			//   			minCost = newCost;
			//   			centers.set(i, point) // imposta come nuovo medoide il punto corrente
			//
			// converged = true
			// for(i=0 to k)
			//   * controlla che la nuova distanza dei nuovi medoidi dai vecchi sia > epsilon*epsilon *
			//   * oppure si può semplicemente controllare se il vecchio costo sia migliore di un certo epsilon rispetto al vecchio *





			// converged = true;
			// for(int i=0; i<medoids.size(); i++)
			// {
			// 	VectorWithNorm newCenter = medoids.get(i);
			// 	if (converged && cosineDistance(newCenter, centers.get(i)) > epsilon*epsilon)
			// 		converged = false;
			// 	centers.set(i, newCenter);
			// }

			cost = costAccum.value();
			iteration++;
		}
		System.out.println("KMedoids converged in "+iteration+" iterations with cost "+cost);
	}

	/*private List<VectorWithNorm> initRandom(JavaRDD<VectorWithNorm> data)
	{
		return data.takeSample(false, k, new XORShiftRandom(seed).nextInt());	
	}*/

	private List<VectorWithNorm> initRandom(JavaRDD<VectorWithNorm> data)
	{
		List<VectorWithNorm> list = data.takeSample(false, k, new XORShiftRandom(seed).nextInt());
		for (int i=0; i<list.size(); i++)
			list.get(i).setCenterIndex(i);
		return list;
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

	private VectorWithNorm computeMedoid(List<VectorWithNorm> vectors){
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		for(int i=0; i<vectors.size(); i++) {
			double tempDistance = 0.0;
			for(int j=0; j<vectors.size(); j++)
				if (i!=j)
					tempDistance += cosineDistance(vectors.get(i),vectors.get(j));
			if(tempDistance < bestDistance){
				bestDistance = tempDistance;
				bestIndex = i;
			}
		}
		return vectors.get(bestIndex);
	}

	private double fastSquaredDistance(VectorWithNorm v1, VectorWithNorm v2){
		return Vectors.sqdist(v1.vector, v2.vector);
	}

	private Tuple2<Integer, Double> findClosestCosine(List<VectorWithNorm> centers, VectorWithNorm point)
	{
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		int i = 0;
		for(VectorWithNorm center : centers)
		{
			double distance = cosineDistance(center, point);
			if(distance < bestDistance)
			{
				bestDistance = distance;
				bestIndex = i;
			}
			i++;
		}
		point.setCenterIndex(bestIndex);
		return new Tuple2<>(bestIndex, bestDistance);
	}

	private static double cosineDistance(VectorWithNorm v1, VectorWithNorm v2) {
	    double cosine = BLAS.dot(v1.vector, v2.vector) / (v1.norm * v2.norm);
	    if (cosine > 1.0) {
	        // Mathematically, this should't be possible, but due to the
	        // propagation of errors in floating point operations, it
	        // happens
	        cosine = 1;
	    }
	    // If you wish to use this function with vectors that can have
	    // negative components (like the ones given by word2vec), then
	    // rescale by PI instead of PI/2
	    return (2 / Math.PI) * Math.acos(cosine);
	  }
}
