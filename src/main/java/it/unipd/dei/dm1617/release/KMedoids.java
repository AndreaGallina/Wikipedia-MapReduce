package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.util.random.XORShiftRandom;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.lang.Math;
import java.io.Serializable;

/**
* Class implementing all the necessary methods to perform the k-medoids clustering using cosine
* distance.
*/
public class KMedoids implements Serializable {

	private double epsilon = 1e-4;	// Convergeance threshold.
	private int maxIterations;		  // Max # of iterations.
	private int k; 								  // Number of clusters.
	private long seed;							// Seed of randomness.
	List<List<VectorWithNorm>> finalPartition;	// List containing the final clustering structure for
																							// printing purposes.
	boolean returnFinalClustering = false;		  // Whether to return the final clustering.
	double bestCost; // Best cost function obtained.

	public KMedoids(int k, int maxIterations, long seed, boolean returnFinalClustering) {
		this.k = k;
		this.maxIterations = maxIterations;
		this.seed = seed;
		this.returnFinalClustering = returnFinalClustering;
		bestCost = Double.POSITIVE_INFINITY;
		if(returnFinalClustering) {
			finalPartition = new ArrayList<List<VectorWithNorm>>(k);
			for(int i=0; i<k; i++) {
				finalPartition.add(new ArrayList<VectorWithNorm>());
			}
		}
	}

	public int getK() {
		return k;
	}

	/**
	 * Prepares the input data and runs the clustering algorithm.
	 * @param data [JavaRDD containing the dataset of documents]
	 * @param sc   [Spark Context needed to setup some Spark variables (accumulator, broadcast etc.)]
	 */
	public void run(JavaRDD<Vector> data, JavaSparkContext sc) {
		// Filters out the vectors with all components set to zero, since the cosine distance between
		// a zero-vector and an arbirtrary vector yields an infinite value.
		JavaRDD<Vector> filteredData = data.filter((v) -> v.numNonzeros()>0);

		// Computes and caches the norms of the input vectors.
		JavaRDD<Double> norms = filteredData.map((v) -> Vectors.norm(v, 2.0));
		norms.cache();
		// Maps the vectors with norm to a JavaRDD.
		JavaRDD<VectorWithNorm> zippedData = filteredData.zip(norms).map((tuple) ->
			new VectorWithNorm(tuple._1,tuple._2));

		// Creates an RDD where all the duplicate vectors are filtered out, since the
		// initial K medoids chosen at random must be distinct.
		// This will be used only for the initialization of the medoids.
		JavaRDD<Vector> filteredDistinctData = filteredData.distinct();
		JavaRDD<Double> distinctNorms = filteredDistinctData.map((v) -> Vectors.norm(v, 2.0));
		JavaRDD<VectorWithNorm> zippedDistinctData = filteredDistinctData.zip(distinctNorms)
			.map((tuple) -> new VectorWithNorm(tuple._1,tuple._2));
		
		// Runs the clustering algorithm.
		kMedoids(zippedData,zippedDistinctData,sc);
	}

	/**
	 * Performs the k-medoids clustering of the input dataset.
	 * @param data [JavaRDD containing the dataset of documents]
	 * @param sc   [Spark Context needed to setup some Spark variables (accumulator, broadcast etc.)]
	 */
	private void kMedoids(JavaRDD<VectorWithNorm> data, JavaRDD<VectorWithNorm> distinctData,
												JavaSparkContext sc) {
		// Chooses a list of k random distinct medoids.
	  List<VectorWithNorm> centers = new ArrayList<>(initRandom(distinctData));

	  // Warns the user that the data has less than K distinct points, hence the algorithm will
	  // return less than K clusters.
	  if(centers.size()<k) {
	  	System.out.println("Could not find k= " + k + " distinct medoids; the algorithm will only "
				+ "return k= " + centers.size() +" clusters.");
	  }

	  boolean converged = false;
	  double cost = Double.POSITIVE_INFINITY; // Clustering cost calculated up to the i-th iteration.
	  int iteration = 0;

		// Iterates until the maximum number of iterations has been reached or the objective
		// function converges.
	  while(iteration < maxIterations && !converged) {
	    // Objective function cost accumulator.
	    DoubleAccumulator costAccum = sc.sc().doubleAccumulator();

	    // Broadcasts the center list to all the workers.
	    Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);
		
			// Partitions the points into clusters. The key field of the map represents the index
			// of the cluster, while the value field contains the points of the cluster.
	    JavaPairRDD<Integer, List<VectorWithNorm>> clustersMapRDD =
	    	data.mapPartitionsToPair((points) -> {
					List<VectorWithNorm> thisCenters = bcCenters.value();

		      // Number of elements of a vector representing a document (equal to "vocabSize").
		      int dims = thisCenters.get(0).vector.size(); 
				
					// Data structure that will contain the points divided into clusters.
		      List<List<VectorWithNorm>> thisPartition = new ArrayList<List<VectorWithNorm>>(k);
		      for(int i=0; i<k; i++) {
		        thisPartition.add(new ArrayList<VectorWithNorm>());
		      }
				
					// Adds each input point to its respective cluster.
		      while(points.hasNext()) {
		        VectorWithNorm point = points.next();

		      	// Finds the closest medoid to the input point and its distance.
		        Tuple2<Integer, Double> bestPartition = findClosestMedoid(thisCenters, point);	                    

		        // Increases the objective function cost by the squared distance between the point and
		        // its cluster's medoid.
		        costAccum.add(bestPartition._2);
		              
		        // Adds the point to its cluster.
		        thisPartition.get(bestPartition._1).add(point);
		      }
				
		      List<Tuple2<Integer, List<VectorWithNorm>>> partitions = new ArrayList<>();

		      // For each cluster, initializes a tuple containing the index of the cluster and
		      // the list of the points belonging to that cluster.
		      for(int i=0; i<k; i++) {
		        if(thisPartition.get(i).size()>0) {
		          partitions.add(new Tuple2<>(i, thisPartition.get(i)));
		        }
		      }

		      return partitions.iterator();
		          
		    }).reduceByKey((tuple1, tuple2) -> {
		      // Merges all the points belonging to one cluster assigned by the different workers.
		      List<VectorWithNorm> clusterPoints = new ArrayList<VectorWithNorm>();
		      clusterPoints.addAll(tuple1);
		      clusterPoints.addAll(tuple2);

		      return clusterPoints;
		    });

	    // Returns the resulting clustering in a Map where keys correspond to the cluster indices
	    // and the values correspond to the points belonging to that cluster. 
	    Map<Integer, List<VectorWithNorm>> clustersMap = clustersMapRDD.collectAsMap();

			// Destroy the broadcast variable.
			bcCenters.destroy();
			    

			long start = System.nanoTime();

			// Computes a new medoid for each cluster. Each medoid becomes the new center for its cluster.
			List<Tuple2<Integer,VectorWithNorm>> calcNewMedoids =
				clustersMapRDD.mapPartitions((clusters) -> {
	      	List<Tuple2<Integer,VectorWithNorm>> newMedoids = new ArrayList<>();
	      	while(clusters.hasNext()) {
	        	Tuple2<Integer, List<VectorWithNorm>> cluster = clusters.next();
	        	int i = cluster._1();
	          List<VectorWithNorm> points = cluster._2();
	          VectorWithNorm newMedoid = computeMedoid(points);
	          newMedoids.add(new Tuple2<>(i, newMedoid));
	      	}
	      	return newMedoids.iterator();
	      }).collect();

			// Set the new medoids as centers.
      for(Tuple2<Integer, VectorWithNorm> newMedoid : calcNewMedoids) {
      	centers.set(newMedoid._1, newMedoid._2);
      }

			double finish = (System.nanoTime() - start) / 1e9;

			// If the difference between the cost computed in this iteration (costAccum) and 
			// the one computed during the previous iteration is not greater than the specified 
			// threshold, then the objective function still has not converged.
			// This also ensures that the algorithm stops if the cost function increased during 
			// the current iteration. 
			if (cost-costAccum.value() < epsilon) {
			  converged = true;
			}

		  // If the user wants the final clustering, add each point to the finalPartition
		  // We also make sure that the cost in the current iteration hasn't increased,
		  // in which case we simply keep the cluster obtained from the previous iteration.
		  if(returnFinalClustering && cost>costAccum.value()) { 
		    for(Map.Entry<Integer, List<VectorWithNorm>> entry : clustersMap.entrySet()) {
        	int i = entry.getKey();
        	List<VectorWithNorm> points = entry.getValue();
        	finalPartition.get(i).clear(); // Clears the result of the previous iteration.
        	finalPartition.get(i).addAll(points);
        }
		  }
	    
			// Updates the cost variables.
			bestCost = cost > costAccum.value() ? costAccum.value() : cost;
		  cost = costAccum.value();
		  iteration++;
	  }

	  System.out.println("KMedoids converged in " + iteration + " iterations with cost " + bestCost);
	}


	/**
	* Randomly chooses k points from the input data as cluster medoids.
	* @param  data [JavaRDD containing the dataset of documents]
	* @return      [List of VectorWithNorm containing the randomly chosen medoids]
	*/
	private List<VectorWithNorm> initRandom(JavaRDD<VectorWithNorm> data) {
		return data.takeSample(false, k, new XORShiftRandom(seed).nextInt());
	}

	/**
	 * Computes the medoid (the point that minimizes the sum of distances to the other points of
	 * the cluster) of the input vectors.
	 * @param  vectors [Input list of vector]
	 * @return         [Vector chosen as medoid]
	 */
	private VectorWithNorm computeMedoid(List<VectorWithNorm> vectors) {
		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		for(int i=0; i<vectors.size(); i++) {
			double tempDistance = 0.0;
			for(int j=0; j<vectors.size(); j++) {
				if (i!=j) {
					tempDistance += cosineDistance(vectors.get(i),vectors.get(j));
				}
			}
			if(tempDistance < bestDistance) {
				bestDistance = tempDistance;
				bestIndex = i;
			}
		}
		return vectors.get(bestIndex);
	}

	/**
	 * Finds the closest medoid to the input vector using cosine distance.
	 * @param  medoids [List of medoids]
	 * @param  point   [Vector to be clustered]
	 * @return         [Tuple2 containing the index of the closest medoid and the distance between the
	 * 									two vectors]
	 */
	private Tuple2<Integer, Double> findClosestMedoid(List<VectorWithNorm> medoids,
																										VectorWithNorm point) {

		double bestDistance = Double.POSITIVE_INFINITY;
		int bestIndex = 0;
		int i = 0;
		boolean isMedoid = false;

		// Checks if the point is a medoid, so as to skip useless calculations 
		// and directly return the index of the medoid and a zero cost. 
		// Omitting this check would yield an incorrect clustering, since the cosine distance
		// between some medoids might be zero even though they are not equal, with the risk
		// for the medoid to be assigned to another cluster.
		for(VectorWithNorm medoid : medoids) {
			if(point.vector.equals(medoid.vector)) {
				isMedoid = true;
				bestIndex = i;
				bestDistance = 0.0;
				break;
			}
			i++;
		}

		// If the point is not a medoid then proceed to find the closest medoid.
		if(!isMedoid) {
			i = 0;		
			for(VectorWithNorm medoid : medoids) {
				double distance = cosineDistance(medoid, point);
				if(distance < bestDistance) {
					bestDistance = distance;
					bestIndex = i;
				}
				i++;
			}
		}
		return new Tuple2<>(bestIndex, bestDistance);
	}

	/**
	 * Returns the cosine distance between the input vectors.
	 * @param  v1 [First input vector]
	 * @param  v2 [Second input vector]
	 * @return    [Cosine distance between the two vectors]
	 */
	private static double cosineDistance(VectorWithNorm v1, VectorWithNorm v2) {
	    double cosine = BLAS.dot(v1.vector, v2.vector) / (v1.norm * v2.norm);
	    if (cosine > 1.0) {
	      cosine = 1;
	    }
	    return (2 / Math.PI) * Math.acos(cosine);
	}
}