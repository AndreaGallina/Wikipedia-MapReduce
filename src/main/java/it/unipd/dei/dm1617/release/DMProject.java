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
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Set;

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
     * pages belonging to each cluster
     * @param lemmatizedPages [JavaRDD of wikipedia pages]
     * @param tfidf           [TF-IDF transformation of the pages]
     * @param kmed            [The trained K-Medoids model]
     */
    public static void printClustering(JavaRDD<WikiPage> lemmatizedPages, JavaRDD tfidf, KMedoids kmed){
        System.out.println("\n\n------- RESULTING CLUSTERING --------");

        JavaPairRDD<WikiPage, Vector> pagesAndVectors = lemmatizedPages.zip(tfidf);

        List<Tuple2<WikiPage, Vector>> listPagesAndVectors = new ArrayList<Tuple2<WikiPage, Vector>>(pagesAndVectors.collect());

        for(int i = 0; i<kmed.getK(); i++)
        {
            // Since each tfidf vector does not have a 1 to 1 correspondence to each WikiPage,
            // we must use the java collection "Set" which doesn't allow duplicates
            Set<String> set1 = new HashSet();
            System.out.println("\n\nK = "+i+ "");
            for(int j = 0; j< kmed.finalPartition.get(i).size(); j++)
            {
                for(Tuple2<WikiPage, Vector> tuple : listPagesAndVectors)
                {
                    // We print the title only if it corresponds to a wikipedia page AND if the corresponding title
                    // has not been printed yet (the method .add(E) of Set returns true only if the element has not yet been added to the set).
                    if(tuple._2.equals(kmed.finalPartition.get(i).get(j).vector) && set1.add(tuple._1.getTitle()))
                        System.out.println(tuple._1.getTitle());
                }            
            }
        }
    }

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
    // -------------------------------------------------------//
    
    // ---------- COMMENT THIS ON FIRST RUN ----------------- //
    // Reads lemmatized wikipedia pages.
    JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "small-dataset-lemmas").cache();
    // -------------------------------------------------------//

    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();


    // Transform the sequence of lemmas in vectors of counts in a space of the specified number of
    // dimensions, using the said number of top lemmas as the vocabulary.
    JavaRDD<Vector> tf = new CountVectorizer()
      .setVocabularySize(100)
      .transform(lemmas)
      .cache();

    // Converts the data in a TF-IDF representation.
    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);


    // Sets the number of cluster and maximum iterations.
    int[] numClusters = {900};
    int maxIterations = 1000;

    // In case we need to print the final clustering
    boolean returnFinalClustering = false;
    
    // Computes k-median clustering of the TF-IDF dataset.
    for(int k : numClusters) {
        long start = System.nanoTime();
        KMedoids kmed = new KMedoids(k, maxIterations, 1000L, returnFinalClustering);
        kmed.run(tfidf, sc);
        double finish = (System.nanoTime() - start) / 1e9;
        System.out.println("Done for k="+k+" in "+finish);
        
        if(returnFinalClustering)
            printClustering(lemmatizedPages, tfidf, kmed);
    }

  }

}
