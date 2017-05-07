package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import it.unipd.dei.dm1617.examples.Partition;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.DenseVector;
import scala.Tuple2;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;


/**
 * Example program to show how to partition a dataset into clusters.
 * Make sure that small-dataset-lemmas has already been computed.
 */
public class PartitioningTester {

  public static void main(String[] args) {
    // Removes all the infodump shown on terminal by spark.
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Usual setup.
    SparkConf conf = new SparkConf(true).setAppName("DMProject");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Reads lemmatized wikipedia pages

    JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "small-dataset-lemmas").cache();

    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();


    JavaRDD<Vector> tf = new CountVectorizer()
      .setVocabularySize(100)
      .transform(lemmas)
      .cache();

    JavaRDD<Vector> tfidf = new IDF()
      .fit(tf)
      .transform(tf);

    // Randomly extracts 2 documents to work as cluster centers. Remove the
    // third parameter to unfix the seed. Fixing the seed ensures that the
    // same elements are always chosen during subsequent runs of the program,
    // which is useful for testing purposes.
    List<Vector> randomCenters = tfidf.takeSample(false, 2, 100);
    for(int i=0; i<randomCenters.size(); i++)
        System.out.println("Chosen center #" + i + ": " + randomCenters.get(i));

    // Randomly extracts 2 documents to later assign to a cluster.
    // This operation is done for testing purposes to limit the size of the
    // input. In the actual application, tfidf (the whole small dataset) will
    // be used directly.
    List<Vector> randomDocuments = tfidf.takeSample(false, 2, 10L);
    for(int i=0; i<randomCenters.size(); i++)
        System.out.println("Chosen document #" + i + ": " + randomDocuments.get(i));

    // Parallelizes the randomDocuments. Again, this operation is necessary for
    // testing purposes as the partitioning function works with RDDs.
    JavaRDD<Vector> dRandomDocuments = sc.parallelize(randomDocuments, sc.defaultParallelism());

    // Partitions the documents into clusters.
    JavaRDD<Tuple2<Vector, Vector>> dPartitions = Partition.documentPartitioning(dRandomDocuments,randomCenters);

    List<Tuple2<Vector, Vector>> partitions = dPartitions.collect();
    for(int i=0; i<partitions.size(); i++)
        System.out.println("Center of " + i + ": " + partitions.get(i)._2());


    // Uncomment the next line and comment the previous line to run the
    // partitioning on the whole dataset.
    // Partition.documentPartitioning(tfidf,randomCenters);

  }

}
