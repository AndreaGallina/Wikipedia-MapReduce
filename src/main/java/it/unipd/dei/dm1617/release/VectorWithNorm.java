package it.unipd.dei.dm1617.release;

import it.unipd.dei.dm1617.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.Serializable;

/**
 * Class representing a vector and its norm.
 */
public class VectorWithNorm implements Serializable {
	Vector vector;
	double norm;

	public VectorWithNorm(Vector vec, double norm) {
		this.vector = vec;
		this.norm = norm;
	}
	
	public VectorWithNorm(Vector vec) {
		this(vec, Vectors.norm(vec, 2.0));
	}
}