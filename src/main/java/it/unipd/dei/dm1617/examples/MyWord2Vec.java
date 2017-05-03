package it.unipd.dei.dm1617.examples;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.Tuple2;
import scala.collection.immutable.Map;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

/**
 * Example program to show the basic usage of some Spark utilities.
 */
public class MyWord2Vec {

  public static void main(String[] args) {
    String dataPath = args[0];

    // Usual setup
    SparkConf conf = new SparkConf(true).setAppName("Word2Vec transformation");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load dataset of pages
    JavaRDD<WikiPage> pages = InputOutput.read(sc, dataPath);

    JavaRDD<WikiPage> lemmatizedPages = InputOutput.read(sc, "small-dataset-lemmas").cache();

    JavaRDD<ArrayList<String>> lemmas = lemmatizedPages.map((p) -> {
        return new ArrayList<String>(Arrays.asList(p.getText().split(" ")));
    }).cache();

    Word2VecModel word2VecModel = new Word2Vec().fit(lemmas);


    Map word2Vec = word2VecModel.getVectors();


    // Da qui non so come leggere 

  }

}
