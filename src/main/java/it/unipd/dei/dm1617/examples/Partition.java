package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import java.lang.Math;

import java.io.Serializable;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

public class Partition {

	// Class representing a word.
	private static class Word implements Serializable {

		// The word this class is representing.
		private String name;

		// An arbitrary number, think of it as a single coordinate.
		private double number;

		// A list of coordinates, to simulate working with multidimensional vectors.
		private List<Double> coordinates;

		// The center (represented by another word) of the cluster this word belongs to.
		private Word center;

		public Word(String name, double number) {
			this.name = name;
			this.number = number;
		}

		public Word(String name, List<Double> coordinates) {
			this.name = name;
			this.coordinates = coordinates;
		}

		public void printCoordinates() {
			System.out.println("Printing coordinates for " + name);
			for (int i=0; i<coordinates.size(); i++)
				System.out.println("Coordinate #" + i + ": " + coordinates.get(i));
		}

		public String getName() {
			return name;
		}

		public double getNumber() {
			return number;
		}

		public Word getCenter() {
			return center;
		}

		public List<Double> getCoordinates() {
			return coordinates;
		}

		public void setCenter(Word center) {
			this.center = center;
		}
	}

	private static class Document implements Serializable {

		private Vector tfidf;
		private Vector center;

		public Document(Vector tfidf) {
			this.tfidf = tfidf;
		}

		public Document(Vector tfidf, Vector center) {
			this.tfidf = tfidf;
			this.center = center;
		}

		public Vector getDocumentVector() {
			return tfidf;
		}

		public Vector getCenter() {
			return center;
		}

		public void setCenter(Vector center) {
			this.center = center;
		}
	}

	// Returns the dot product between the input Word vectors.
	public static double componentWiseProduct(Word w1, Word w2) {
		List<Double> coordinates1 = w1.getCoordinates();
		List<Double> coordinates2 = w2.getCoordinates();
		double product = 0;
		int size = coordinates1.size();
		for (int i=0; i<size; i++)
			product += coordinates1.get(i)*coordinates2.get(i);
		//System.out.println("Product between " + w1.getName() + " and " + w2.getName() + ": " + product);
		return product;
	}

	// Returns the dot product between the input double vectors.
	public static double componentWiseProduct(double[] v1, double[] v2) {
		double product = 0;
		int size = v1.length;
		for (int i=0; i<size; i++)
			product += v1[i]*v2[i];
			//System.out.println("Product between " + w1.getName() + " and " + w2.getName() + ": " + product);
		return product;
	}

	// Returns the norm of the input Word vector.
	public static double norm(Word w1) {
		List<Double> coordinates = w1.getCoordinates();
		double norm = 0;
		int size = coordinates.size();
		for (int i=0; i<size; i++)
			norm += Math.pow((coordinates.get(i)),2);
		//System.out.println("Squared norm of " + w1.getName() + ": " + norm);
		return Math.sqrt(norm);
	}

	// Returns the norm of the input double vector.
	public static double norm(double[] v1) {
		double norm = 0;
		int size = v1.length;
		for (int i=0; i<size; i++)
			norm += Math.pow(v1[i],2);
		//System.out.println("Squared norm of " + w1.getName() + ": " + norm);
		return Math.sqrt(norm);
	}

	// Returns the euclidean distance between two input Word vectors.
	public static double euclideanDistance(Word w1, Word w2) {
		return Math.sqrt(Math.pow((w1.getNumber()-w2.getNumber()),2));
	}

	// Returns the cosine distance between two input Word vectors.
	public static double cosineDistance(Word w1, Word w2) {
		return Math.acos(componentWiseProduct(w1,w2)/(norm(w1)*norm(w2)))/(Math.PI/2);
	}

	// Returns the cosine distance between two input double vectors.
	public static double cosineDistance(double[] v1, double[] v2) {
		return Math.acos(componentWiseProduct(v1,v2)/(norm(v1)*norm(v2)))/(Math.PI/2);
	}

	// Partitions a mockup dataset with points belonging to a multidimensional metric space.
	public static void multiDimensionalPartitioning() {
	    // Usual setup.
	    SparkConf conf = new SparkConf(true).setAppName("Partition");
	    JavaSparkContext sc = new JavaSparkContext(conf);

	    // Initializes some words with randomly generated coordinates.
	    List<Word> numbers = new ArrayList<>(5);
	    for (int i=0; i<5; i++) {
	    	List<Double> coordinates = new ArrayList<>(2);
	    	for (int j=0; j<2; j++)
	    		coordinates.add(100.0*Math.random());
	    	numbers.add(new Word(""+(i+1)+"",coordinates));
	    	numbers.get(i).printCoordinates();
	    }

	    // Initializes some centers.
	    List<Word> centers = new ArrayList<>(2);
	    List<Double> coordinates = new ArrayList<>(2);
	    coordinates.add(new Double(40));
	    coordinates.add(new Double(50));
	    centers.add(new Word("Center 1",coordinates));
	  	centers.get(0).printCoordinates();
	    coordinates = new ArrayList<>(2);
	    coordinates.add(new Double(60));
	    coordinates.add(new Double(70));
	    centers.add(new Word("Center 2",coordinates));
	    centers.get(1).printCoordinates();

    	int numPartitions = sc.defaultParallelism();
   		System.err.println("Splitting data in " + numPartitions + " partitions");

   		// Splits the work among numPartitions partitions.
   		JavaRDD<Word> dWords = sc.parallelize(numbers, numPartitions);

   		// Assigns a center to each point using cosine distance (see Partition algorithm on the powerpoint).
   		JavaRDD<Word> dPartitions = dWords.map((w) -> {
   			System.out.println("Computing closest center for " + w.getName());
   			double minDist = cosineDistance(w,centers.get(0));
   			//System.out.println("Cosine distance between " + w.getName() + " and " + centers.get(0).getName() + ": " + minDist);
   			int closestCenter = 0;
   			for (int i=1; i<centers.size(); i++) {
   				double tempDist = cosineDistance(w,centers.get(i));
   				//System.out.println("Cosine distance between " + w.getName() + " and " + centers.get(i).getName() + ": " + tempDist);
   				if (minDist > tempDist) {
   					minDist = tempDist;
   					closestCenter = i;
   				}
   			}
   			w.setCenter(centers.get(closestCenter));
   		    //System.out.println("Cosine distance between " + w.getName() + " and its closest center " + centers.get(closestCenter).getName() + ": " + minDist);
   			return w;
   		});

   		// Collects the result into a single list and prints the chosen centers.
   		List<Word> partitions = dPartitions.collect();
   		for(int i=0; i<partitions.size(); i++)
   			System.out.println("Center of " + (i+1) + ": " + partitions.get(i).getCenter().getName());

	    System.out.println("Multidimensional partitioning completed");
	}

	// Partitions the input document collection into clusters whose centers are represented by the centers vector.
	public static void documentPartitioning(JavaRDD<Vector> tfidf, List<Vector> centers) {
	    JavaRDD<Document> dPartitions = tfidf.map((v) -> {
	    	System.out.println("Computing closest center");
   			double minDist = cosineDistance(v.toDense().values(),centers.get(0).toDense().values());
   			System.out.println("Cosine distance between vector and center #0: " + minDist);
   			int closestCenter = 0;
   			for (int i=1; i<centers.size(); i++) {
   				double tempDist = cosineDistance(v.toDense().values(),centers.get(i).toDense().values());
   				System.out.println("Cosine distance between vector and center #" + i + " : " + tempDist);
   				if (minDist > tempDist) {
   					minDist = tempDist;
   					closestCenter = i;
   				}
   			}
   			Document d = new Document(v,centers.get(closestCenter));
   		    System.out.println("Cosine distance between vector and its closest center: " + minDist);
   			return d;
	    });

	    List<Document> partitions = dPartitions.collect();
   		for(int i=0; i<partitions.size(); i++)
   			System.out.println("Center of " + i + ": " + partitions.get(i).getCenter());

	    System.out.println("Document partitioning completed");
	}

	// Partitions a mockup dataset with points belonging to a one-dimensional metric space. Test method, do not mind it.
	public static void oneDimensionalPartitioning() {
		// Usual setup
	    SparkConf conf = new SparkConf(true).setAppName("Partition");
	    JavaSparkContext sc = new JavaSparkContext(conf);

	    List<Word> numbers = new ArrayList<>(10);
	    for (int i=0; i<10; i++)
	    	numbers.add(new Word(""+(i+1)+"",i+1));
	    List<Word> centers = new ArrayList<>(2);
	    centers.add(new Word("3",3));
	    centers.add(new Word("7",7));

	    // Sequential (local) partitioning for testing purposes.
	    /*for(int i=0; i<numbers.size(); i++) {
	    	double dist1 = euclideanDistance(numbers.get(i),centers.get(0));
	    	double dist2 = euclideanDistance(numbers.get(i),centers.get(1));
	    	System.out.println("Distanza tra " + (i+1) + " e " + "3: " + dist1);
	    	System.out.println("Distanza tra " + (i+1) + " e " + "7: " + dist2);
	    	if (dist1 < dist2)
	    		numbers.get(i).setCenter(centers.get(0));
	    	else
	    		numbers.get(i).setCenter(centers.get(1));
	    	System.out.println("Centro di " + (i+1) + ": " + numbers.get(i).getCenter().getName());
	    } */

    	int numPartitions = sc.defaultParallelism();
   		System.err.println("Splitting data in " + numPartitions + " partitions");

   		JavaRDD<Word> dWords = sc.parallelize(numbers, numPartitions);

   		JavaRDD<Word> dPartitions = dWords.map((w) -> {
   			double minDist = euclideanDistance(w,centers.get(0));
   			int closestCenter = 0;
   			System.out.println("Computing closest center for " + w.getName());
   			for (int i=1; i<centers.size(); i++) {
   				double tempDist = euclideanDistance(w,centers.get(i));
   				if (minDist > tempDist) {
   					minDist = tempDist;
   					closestCenter = i;
   				}
   			}
   			w.setCenter(centers.get(closestCenter));
   			return w;
   		});

   		List<Word> partitions = dPartitions.collect();
   		for(int i=0; i<partitions.size(); i++)
   			System.out.println("Center of " + (i+1) + ": " + partitions.get(i).getCenter().getName());

	    System.out.println("One-dimensional partitioning completed");
	}

	public static void main(String[] args) {
		// Removes all the infodump shown on terminal by spark
	    Logger.getLogger("org").setLevel(Level.OFF);
	    Logger.getLogger("akka").setLevel(Level.OFF);

	    if(args.length != 0)
			if(Integer.parseInt(args[0]) == 0) {
				System.out.println("One-dimensional partitioning...");
				oneDimensionalPartitioning();
			} else if (Integer.parseInt(args[0]) == 1) {
				System.out.println("Multidimensional partitioning...");
				multiDimensionalPartitioning();
			}
	}

}