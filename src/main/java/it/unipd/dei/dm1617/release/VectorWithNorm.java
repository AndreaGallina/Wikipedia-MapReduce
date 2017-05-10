package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.broadcast.Broadcast;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

public class VectorWithNorm implements Serializable{
	Vector vector;
	double norm;
	int centerIndex;

	public VectorWithNorm(Vector vec, double norm){
		this.vector = vec;
		this.norm = norm;
	}
	public VectorWithNorm(Vector vec){
		this(vec, Vectors.norm(vec, 2.0));
	}

	public void setCenterIndex(int index){
		this.centerIndex = index;
	}
}