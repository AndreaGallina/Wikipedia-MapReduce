package it.unipd.dei.dm1617;


import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

public class Distance {

  /**
   * Cosine distance between vectors where all the elements are positive.
   */
  public static double cosineDistance(Vector a, Vector b) {
    if (a.size() != b.size()) {
      throw new IllegalArgumentException("Vectors should be in the same space");
    }
    double num = 0;
    for (int i=0; i<a.size(); i++) {
      num += a.apply(i) * b.apply(i);
    }
    double normA = Vectors.norm(a, 2);
    double normB = Vectors.norm(b, 2);

    double cosine = num / (normA * normB);
    if (cosine > 1.0) {
      // Mathematically, this should't be possible, but due to the
      // propagation of errors in floating point operations, it
      // happens
      cosine = 1;
    }
    // If you wish to use this function with vectors that can have
    // negative components (like the ones given by word2vec), then
    // rescale by PI instead of PI/2
    return (2 / Math.PI) * Math.acos(cosine);
  }
    
}
