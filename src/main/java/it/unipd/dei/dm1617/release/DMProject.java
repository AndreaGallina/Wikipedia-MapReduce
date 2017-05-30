package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
import java.util.Scanner;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

/**
 * Main class managing the input dataset and the k-median clustering.
 * Reads the input data, lemmatizing it and representing it in TF-IDF format and runs the
 * clustering algorithm.
 */
public class DMProject {

  /**
   * Prints a list of the resulting clustering, showing the title of the wikipedia
   * pages belonging to each cluster.
   * @param lemmatizedPages [JavaRDD of wikipedia pages]
   * @param tfidf           [TF-IDF transformation of the pages]
   * @param kmed            [The trained K-Medoids model]
   */
  public static void printClustering(JavaRDD<WikiPage> lemmatizedPages, JavaRDD tfidf,
                                     KMedoids kmed) {
    System.out.println("\n\n------- RESULTING CLUSTERING --------");

    JavaPairRDD<WikiPage, Vector> pagesAndVectors = lemmatizedPages.zip(tfidf);

    List<Tuple2<WikiPage, Vector>> listPagesAndVectors
        = new ArrayList<Tuple2<WikiPage, Vector>>(pagesAndVectors.collect());

    for(int i = 0; i<kmed.getK(); i++) {
      // Since each tfidf vector does not have a 1 to 1 correspondence to each WikiPage,
      // a java collection "Set" which does not allow duplicates must be used.
      Set<String> set1 = new HashSet();
      System.out.println("\n\nK = "+i+ "");
      for(int j = 0; j< kmed.finalPartition.get(i).size(); j++) {
        for(Tuple2<WikiPage, Vector> tuple : listPagesAndVectors) {
          // The title is printed only if it corresponds to a wikipedia page AND if the
          // corresponding title has not been printed yet (the method .add(E) of Set returns true
          // only if the element has not yet been added to the set).
          if(tuple._2.equals(kmed.finalPartition.get(i).get(j).vector)
              && set1.add(tuple._1.getTitle())) {
            System.out.println(tuple._1.getTitle());
          }
        }            
      }
    }
  }

  public static void main(String[] args) {
    // Removes all the infoormation shown on terminal by Spark.
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    int k = 390;                            // Number of clusters
    int vocabularySize = 100;               // Number of elements in each TFIDF vector
    int numPartitions = 4;                  // Number of partitions
    boolean returnFinalClustering = false;  // Whether to print the final clustering.
    int maxIterations = 250;                // Max number of iterations
    long seed = 1000L;                      // Seed of randomness
    String dataPath = args[0];              // Input dataset path

    Scanner scan = new Scanner(System.in);

    // Allows the user to customize several parameters.
    System.out.print("Do you want to use the default parameters? (Y/N): ");
    String res = scan.next();
    if(res.equals("N") || res.equals("n")) {
      System.out.print("Enter the number of clusters (k): ");
      k = scan.nextInt();

      System.out.print("Enter the vocabulary size: ");
      vocabularySize = scan.nextInt();

      System.out.print("Enter the number of partitions: ");
      numPartitions = scan.nextInt();

      System.out.print("Enter the maximum number of iterations: ");
      maxIterations = scan.nextInt();

      System.out.print("Do you want to print the final clustering? (Y/n): ");
      res = scan.next();
      if(res.equals("Y") || res.equals("y")) {
        returnFinalClustering = true;
      }
    }

    // Prints the parameters to be used by the clustering algorithm.
    System.out.println("Clustering run with the following parameters:");
    System.out.println("Number of clusters: " + k);
    System.out.println("Vocabulary size: " + vocabularySize);
    System.out.println("Number of partitions: " + numPartitions);
    System.out.println("Maximum number of iterations: " + maxIterations + "\n");

    // Initializes Spark context.
    SparkConf conf = new SparkConf(true).setAppName("DMProject");
    JavaSparkContext sc = new JavaSparkContext(conf);
    
    // Loads the dataset of pages.
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);
    
    // Lemmatizes the Wikipedia pages.
    JavaRDD<WikiPage> lemmatizedPages = Lemmatizer.lemmatizeWikiPages(pages).cache();
    
    // ----------------------------- //
    // InputOutput.write(lemmatizedPages, "lemmatized-dataset");
    // JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "lemmatized-dataset").cache();
    // ----------------------------- //


    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();

    // Transforms the sequence of lemmas in vectors of counts in a space of the specified number of
    // dimensions, using the said number of top lemmas as the vocabulary.
    JavaRDD<Vector> tf = new CountVectorizer()
      .setVocabularySize(vocabularySize)
      .transform(lemmas)
      .cache();

    // Converts the data in a TF-IDF representation.
    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);

    // Repartitions the dataset into numPartitions partitions.
    tfidf = tfidf.repartition(numPartitions);

    long start = System.nanoTime();
    
    // Computes k-medoids clustering of the TF-IDF dataset.
    KMedoids kmed = new KMedoids(k, maxIterations, seed, returnFinalClustering);
    kmed.run(tfidf, sc);

    double finish = (System.nanoTime() - start) / 1e9;
    System.out.println("Done for k= " + k + " in " + finish + " seconds.");
    
    // Prints the clustering if specified by the user.
    if(returnFinalClustering) {
      printClustering(lemmatizedPages, tfidf, kmed);
    }
  }
}
