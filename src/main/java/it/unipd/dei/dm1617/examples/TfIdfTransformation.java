package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Example program to show the basic usage of some Spark utilities.
 */
public class TfIdfTransformation {

  public static void main(String[] args) {
    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("Tf-Ifd transformation");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load dataset of pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

    // Get text out of pages
    JavaRDD<String> texts = pages.map((p) -> p.getText());

    // Get the lemmas. It's better to cache this RDD since the
    // following operation, lemmatization, will go through it two
    // times.
    JavaRDD<ArrayList<String>> lemmas = Lemmatizer.lemmatize(texts).cache();

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

    // In this last step we "zip" toghether the original pages and
    // their corresponding tfidf vectors. We can perform this
    // operation safely because we did no operation changing the order
    // of pages and vectors within their respective datasets,
    // therefore the first vector corresponds to the first page and so
    // on.
    JavaPairRDD<WikiPage, Vector> pagesAndVectors = pages.zip(tfidf);

    // Finally, we print the distance between the first two pages
    List<Tuple2<WikiPage, Vector>> firstPages = pagesAndVectors.take(2);
    double dist = Distance.cosineDistance(firstPages.get(0)._2(), firstPages.get(1)._2());
    System.out.println("Cosine distance between `" +
      firstPages.get(0)._1().getTitle() + "` and `" +
      firstPages.get(1)._1().getTitle() + "` = " + dist);

  }

}
