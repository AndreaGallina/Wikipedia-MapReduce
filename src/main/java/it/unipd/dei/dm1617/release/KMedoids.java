package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.util.random.XORShiftRandom;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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

<<<<<<< HEAD
	private double eps = 1.0;

	public KMedoids(int k, int maxIterations, long seed){
=======
	public KMedoids(int k, int maxIterations, long seed, boolean returnFinalClustering) {
>>>>>>> kMedoids
		this.k = k;
		this.maxIterations = maxIterations;
		this.seed = seed;
<<<<<<< HEAD

		while ((1.0 + (eps / 2.0)) != 1.0) {
	      eps /= 2.0;
	    }
=======
		this.returnFinalClustering = returnFinalClustering;
		if(returnFinalClustering) {
			finalPartition = new ArrayList<List<VectorWithNorm>>(k);
			for(int i=0; i<k; i++) {
				finalPartition.add(new ArrayList<VectorWithNorm>());
			}
		}
	}

	public int getK() {
		return k;
>>>>>>> kMedoids
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
	  if(centers.size()<k)
	  	System.out.println("Could not find k= " + k + " distinct medoids; the algorithm will only "
				+ "return k= " + centers.size() +" clusters.");

	  boolean converged = false;
	  double cost = Double.POSITIVE_INFINITY;
	  int iteration = 0;

		// Iterates until the maximum number of iterations has been reached or the objective
		// function converges.
	  while(iteration < maxIterations && !converged) {
	    System.out.println("Inizio iterazione " + iteration);
	    // Objective function cost accumulator.
	    DoubleAccumulator costAccum = sc.sc().doubleAccumulator();

	    // Broadcasts the center list to all the workers.
	    Broadcast<List<VectorWithNorm>> bcCenters = sc.broadcast(centers);
		
			// Partitions the points into clusters. The key field of the map represents the index
			// of the cluster, while the value field contains the points of the cluster.
	    Map<Integer, List<VectorWithNorm>> clustersMap = data.mapPartitionsToPair((points) -> {

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

	        // Increases the objective function cost by the squared distance between
	        // the point and its cluster's medoid.
	        costAccum.add(Math.pow(bestPartition._2,2));
	              
	        // Adds the point to its cluster.
	        thisPartition.get(bestPartition._1).add(point);
	      }
			
	      List<Tuple2<Integer, List<VectorWithNorm>>> partitions = new ArrayList<>();

	      // For each cluster, initializes a tuple containing the index of the cluster and
	      // the list of the points belonging to that cluster.
	      for(int i=0; i<k; i++) {
	        if(thisPartition.get(i).size()>0) {
	          partitions.add(new Tuple2<>(i, thisPartition.get(i)));
	        } else {
	          System.out.println("Cluster "+i+" is empty");
	        }
	      }

	      return partitions.iterator();
	          
	    }).reduceByKey((tuple1, tuple2) -> {
	      // Merges all the points belonging to one cluster assigned by the different workers.
	      List<VectorWithNorm> clusterPoints = new ArrayList<VectorWithNorm>();
	      clusterPoints.addAll(tuple1);
	      clusterPoints.addAll(tuple2);

	      return clusterPoints;
	    }).collectAsMap();

			// Destroy the broadcast variable.
			bcCenters.destroy();
			    
			System.out.println("Partizionamento completato iterazione " + iteration);

			System.out.println("Inizio calcolo medoidi iterazione " + iteration);
			long start = System.nanoTime();

			List<Tuple2<Integer, VectorWithNorm>> medoids
				= new ArrayList<Tuple2<Integer, VectorWithNorm>>();

			// Computes a new medoid for each cluster. Each medoid becomes the new center for its cluster.
			for(Map.Entry<Integer, List<VectorWithNorm>> entry : clustersMap.entrySet()) {
			  int i = entry.getKey();
			  List<VectorWithNorm> points = entry.getValue();
			  VectorWithNorm newMedoid = computeMedoid(points);

			  medoids.add(new Tuple2<>(i, newMedoid));
			  centers.set(i, newMedoid);
			}

			double finish = (System.nanoTime() - start) / 1e9;
			System.out.println("Calcolo medoidi completato iterazione " + iteration + " in " + finish);

			// If the difference between the cost computed in this iteration (costAccum) and 
			// the one computed during the previous iteration is not greater than the specified 
			// threshold, then the objective function still has not converged.
			if (Math.abs(cost-costAccum.value()) < epsilon) {
			  converged = true;

			  // If the user wants the final clustering, add each point to the finalPartition
			  if(returnFinalClustering){ 
			    for(Map.Entry<Integer, List<VectorWithNorm>> entry : clustersMap.entrySet()) {
			    	int i = entry.getKey();
			    	List<VectorWithNorm> points = entry.getValue();
			    	finalPartition.get(i).addAll(points);
			    }
			  }
	  	}
	    
			// Updates the cost.
		  cost = costAccum.value();
		  System.out.println("Fine iterazione " + iteration + "; costo:" + cost);
		  iteration++;
	  }
	  System.out.println("KMedoids converged in " + iteration + " iterations with cost " + cost);
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

<<<<<<< HEAD

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
=======
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
>>>>>>> kMedoids
	}

	// Helper method per calcolare il numero di elementi uguali in una lista, non cancellare per il momento
	// public Set<Vector> findDuplicates(List<VectorWithNorm> listContainingDuplicates)
	// { 
	// 	final Set<Vector> setToReturn = new HashSet(); 
	// 	final Set<Vector> set1 = new HashSet();
	// 	int numDuplicates = 0;

	// 	for (VectorWithNorm vec : listContainingDuplicates)
	// 	{
	// 		if (set1.add(vec.vector))
	// 		{
	// 			setToReturn.add(vec.vector);
	// 		}
	// 		else
	// 			numDuplicates++;
	// 	}

	// 	System.out.println(numDuplicates);
	// 	return setToReturn;
	// }
}