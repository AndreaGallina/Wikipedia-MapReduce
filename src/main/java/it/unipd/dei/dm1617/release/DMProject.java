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
 * Example program to show the basic usage of some Spark utilities.
 */
public class DMProject {

  public static void main(String[] args) {
    // Removes all the infodump shown on terminal by spark
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("DMProject");
    JavaSparkContext sc = new JavaSparkContext(conf);

    
    // ---------- UNCOMMENT THIS ON FIRST RUN --------------- //
    // Lemmatize pages and saves them to a file so that we dont have to recompute the
    // lematization everytime
    // 
    // //Load dataset of pages
    // JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);
    // 
    // JavaRDD<WikiPage> lemmatizedPages = Lemmatizer.lemmatizeWikiPages(pages).cache();
    // InputOutput.write(lemmatizedPages, "small-dataset-lemmas");
    
    // Reads lemmatized wikipedia pages
    JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "small-dataset-lemmas").cache();

    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();


    // Transform the sequence of lemmas in vectors of counts in a
    // space of 100 dimensions, using the 100 top lemmas as the vocabulary.
    // This invocation follows a common pattern used in Spark components:
    //
    //  - Build an instance of a configurable object, in this case CountVectorizer.
    //  - Set the parameters of the algorithm implemented by the object
    //  - Invoke the `transform` method on the configured object, yielding
    //  - the transformed dataset.
    //
    // In this case we also cache the dataset because the next step,
    // IDF, will perform two passes over it.
    JavaRDD<Vector> tf = new CountVectorizer()
      .setVocabularySize(100)
      .transform(lemmas)
      .cache();

    // Same as above, here we follow the same pattern, with a small
    // addition. Some of these "configurable" objects configure their
    // internal state by means of an invocation of their `fit` method
    // on a dataset. In this case, the Inverse Document Frequence
    // algorithm needs to know about the term frequencies across the
    // entire input dataset before rescaling the counts of the single
    // vectors, and this is what happens inside the `fit` method invocation.
    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);



    int[] numClusters = {1, 2, 3, 4, 5, 6};
    int maxIterations = 1000;
    
    // Trying different values of K
    for(int k : numClusters)
    {    
        KMeansModel clusters = KMeans.train(tfidf.rdd(), k, maxIterations);

        
        double cost = clusters.computeCost(tfidf.rdd());
        System.out.println("Cost: " + cost);
        System.out.println("");
        System.out.println("");
    }



  }

}
