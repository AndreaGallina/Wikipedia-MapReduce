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

import org.apache.log4j.Logger;
import org.apache.log4j.Level;


/**
 * Main class managing the input dataset and the k-median clustering.
 * Reads the input data, lemmatizing it and representing it in TF-IDF format and runs the
 * clustering algorithm.
 */
public class DMProject {

  public static void main(String[] args) {
    // Removes all the infoormation shown on terminal by Spark.
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    String dataPath = args[0];

    // Initializes Spark context.
    SparkConf conf = new SparkConf(true).setAppName("DMProject");
    JavaSparkContext sc = new JavaSparkContext(conf);

    
    // ---------- UNCOMMENT THIS ON FIRST RUN --------------- //
    // Lemmatize pages and saves them to a file so that we dont have to recompute the
    // lematization every time.
    // 
    // //Load dataset of pages
    // JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);
    // 
    // JavaRDD<WikiPage> lemmatizedPages = Lemmatizer.lemmatizeWikiPages(pages).cache();
    // InputOutput.write(lemmatizedPages, "small-dataset-lemmas");
    
    // Reads lemmatized wikipedia pages.
    JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "small-dataset-lemmas").cache();

    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();


    // Transform the sequence of lemmas in vectors of counts in a space of the specified number of
    // dimensions, using the said number of top lemmas as the vocabulary.
    JavaRDD<Vector> tf = new CountVectorizer()
      .setVocabularySize(300)
      .transform(lemmas)
      .cache();

    // Converts the data in a TF-IDF representation.
    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);

    // Sets the number of cluster and maximum iterations.
    int[] numClusters = {9000};
    int maxIterations = 1000;

    System.out.println("Total points:" + tfidf.count() +"\n\n");
    
    // Computes k-median clustering of the TF-IDF dataset.
    for(int k : numClusters) {
        long start = System.nanoTime();
        KMedoids kmed = new KMedoids(k, maxIterations, 1000L);
        kmed.run(tfidf, sc);
        double finish = (System.nanoTime() - start) / 1e9;
        System.out.println("Done for k="+k+" in "+finish);
    }

  }

}
