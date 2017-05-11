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

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map;
import java.io.Serializable;
import java.lang.Math;

/**
 * Class implementing all the necessary methods to perform the k-medoids clustering using cosine
 * distance.
 */
public class KMedoids implements Serializable{
	private double epsilon = 1e-4;
	private int maxIterations;
	private int k;
	private long seed;
 
	public KMedoids(int k, int maxIterations, long seed) {
		this.k = k;
		this.maxIterations = maxIterations;
		this.seed = seed;
	}

	// Prepares the input data and runs the clustering algorithm.
	public void run(JavaRDD<Vector> data, JavaSparkContext sc) {
		// Computes and caches the norms of the input vectors.
		JavaRDD<Double> norms = data.map((v) -> Vectors.norm(v, 2.0));
		norms.cache();
		// Maps the vectors with norm to a JavaRDD.
		JavaRDD<VectorWithNorm> zippedData = data.zip(norms).map((tuple) -> new VectorWithNorm(tuple._1,tuple._2));
		// Runs the clustering algorithm.
		kMedoids(zippedData,sc);
	}


	// Performs the k-medoids clustering of the input dataset.
	private void kMedoids(JavaRDD<VectorWithNorm> data, JavaSparkContext sc) {
 		// Chooses a list of k random centers.
        List<VectorWithNorm> centers = new ArrayList<>(initRandom(data));
 		
        boolean converged = false;
        double cost = 0.0;
        int iteration = 0;
 		
 		// Iterates until the maximum number of iterations has been reached or the objective
 		// function converges.
        while(iteration < maxIterations && !converged)
        {
            System.out.println("Inizio iterazione " + iteration);
            // Objective function cost accumulator.
            DoubleAccumulator costAccum = sc.sc().doubleAccumulator();
            // Broadcasts the center list to all the workers.
            Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);
 			
 			// Partitions the points into clusters. The key field of the map represents the index
 			// of the cluster, while the value field contains the points of the cluster.
            Map<Integer, List<VectorWithNorm>> totalContribs = data.mapPartitionsToPair((points) -> {
                List<VectorWithNorm> thisCenters = bcCenters.value();
                // Number of elements of a vector representing a document (equal to "vocabSize").
                int dims = thisCenters.get(0).vector.size(); 
 				
 				// Initializes the list of clusters.
                List<List<VectorWithNorm>> thisPartition = new ArrayList<List<VectorWithNorm>>(k);
                for(int i=0; i<k; i++)
                    thisPartition.add(new ArrayList<VectorWithNorm>());
 				
 				// Adds each input point to its respective cluster, ignoring points with all fields
 				// set to zero.
                while(points.hasNext()){
                    VectorWithNorm point = points.next();
                    if(point.vector.numNonzeros()!=0){
                    	// Finds the closest center to the input point and their distance.
	                    Tuple2<Integer, Double> bestPartition = findClosestCosine(thisCenters, point);	                    

	                    // Increases the objective function cost by the squared distance between
	                    // the point and its cluster's center.
	                    costAccum.add(Math.pow(bestPartition._2,2));
	                    
	                    // Adds the point to its cluster.
	                    thisPartition.get(bestPartition._1).add(point);
	                }
                }
 				
                List<Tuple2<Integer, List<VectorWithNorm>>> partitions = new ArrayList<>();
 
                // For each cluster, initializes a tuple containing the index of the cluster and
                // the list of the points belonging to that cluster.
                for(int i=0; i<k; i++) {
                    if(thisPartition.get(i).size()>0)
                        partitions.add(new Tuple2<>(i, thisPartition.get(i)));
                }
 
                return partitions.iterator();
            }).reduceByKey((tuple1, tuple2) -> {
                // Merges all the points belonging to one cluster assigned by the different workers.
                List<VectorWithNorm> clusterPoints = new ArrayList<VectorWithNorm>();
                clusterPoints.addAll(tuple1);
                clusterPoints.addAll(tuple2);
 
                return clusterPoints;
            }).collectAsMap();
            System.out.println("Partizionamento completato iterazione " + iteration);

            // Removes the broadcast variable.
            bcCenters.destroy();
 			
 			System.out.println("Inizio calcolo medoidi iterazione " + iteration);
 			long start = System.nanoTime();
            List<Tuple2<Integer, VectorWithNorm>> medoids = new ArrayList<Tuple2<Integer, VectorWithNorm>>();
 			
 			// Computes the medoid for each cluster. Each medoid becomes the new center for its
 			// cluster.
            for(Map.Entry<Integer, List<VectorWithNorm>> entry : totalContribs.entrySet()) {
                int i = entry.getKey();
                List<VectorWithNorm> points = entry.getValue();
                VectorWithNorm newMedoid = computeMedoid(points);
                medoids.add(new Tuple2<>(i, newMedoid));
                centers.set(i, newMedoid);
            }
            double finish = (System.nanoTime() - start) / 1e9;
            System.out.println("Calcolo medoidi completato iterazione " + iteration + " in " + finish);
 			
 			// If the difference between the current cost (costAccum) and the cost computed during
 			// the previous iteration is not bigger than the specified threshold, then the
 			// objective function still has not converged.
            if (Math.abs(cost-costAccum.value()) < epsilon*epsilon)
                converged = true;

 			// Updates the cost.
            cost = costAccum.value();
            System.out.println("Fine iterazione " + iteration + "; costo:" + cost);
            iteration++;
        }
        System.out.println("KMedoids converged in " + iteration + " iterations with cost " + cost);
    }

    // Randomly chooses k points from the input data as cluster centers.
	private List<VectorWithNorm> initRandom(JavaRDD<VectorWithNorm> data) {
		List<VectorWithNorm> list = data.takeSample(false, k, new XORShiftRandom(seed).nextInt());
		for (int i=0; i<list.size(); i++)
			list.get(i).setCenterIndex(i);
		return list;
	}

	// Computes the medoid (the point that minimizes the sum of distances to the other points of
	// the cluster) of the input vectors.
	private VectorWithNorm computeMedoid(List<VectorWithNorm> vectors) {
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		for(int i=0; i<vectors.size(); i++) {
			double tempDistance = 0.0;
			for(int j=0; j<vectors.size(); j++)
				if (i!=j)
					tempDistance += cosineDistance(vectors.get(i),vectors.get(j));
			if(tempDistance < bestDistance) {
				bestDistance = tempDistance;
				bestIndex = i;
			}
		}
		return vectors.get(bestIndex);
	}

	// Finds the closest center to the input vector using cosine distance.
	private Tuple2<Integer, Double> findClosestCosine(List<VectorWithNorm> centers, VectorWithNorm point) {
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		int i = 0;
		for(VectorWithNorm center : centers) {
			double distance = cosineDistance(center, point);
			if(distance < bestDistance) {
				bestDistance = distance;
				bestIndex = i;
			}
			i++;
		}
		point.setCenterIndex(bestIndex);
		return new Tuple2<>(bestIndex, bestDistance);
	}

	// Returns the cosine distance between the input vectors.
	private static double cosineDistance(VectorWithNorm v1, VectorWithNorm v2) {
	    double cosine = BLAS.dot(v1.vector, v2.vector) / (v1.norm * v2.norm);
	    if (cosine > 1.0)
	        cosine = 1;
	    return (2 / Math.PI) * Math.acos(cosine);
	 }
}