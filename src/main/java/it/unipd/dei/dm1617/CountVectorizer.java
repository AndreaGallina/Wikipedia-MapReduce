package it.unipd.dei.dm1617;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Tool to convert a collection of text documents into a collection of
 * vectors of token counts. Further details on this process are
 * provided here:
 * https://nlp.stanford.edu/IR-book/html/htmledition/term-frequency-and-weighting-1.html
 *
 * This is a basic implementation, which you are welcome to extend
 * with more features, like the ability to set the minimum and maximum
 * document frequency of a word to be included in the vocabulary.
 */
public class CountVectorizer implements Serializable {

  private static class TupleComparator implements Comparator<Tuple2<String, Integer>>, Serializable {
    @Override
    public int compare(Tuple2<String, Integer> t1, Tuple2<String, Integer> t2) {
      return t1._2().compareTo(t2._2());
    }
  }

  private static class IteratorConverter implements Iterator<Tuple2<String, Integer>> {
    Iterator<Object2IntMap.Entry<String>> inner;

    public IteratorConverter(Object2IntOpenHashMap<String> wordCounts) {
       inner = wordCounts.object2IntEntrySet().fastIterator();
    }

    @Override
    public boolean hasNext() {
      return inner.hasNext();
    }

    @Override
    public Tuple2<String, Integer> next() {
      Object2IntMap.Entry<String> e = inner.next();
      return new Tuple2<>(e.getKey(), e.getIntValue());
    }
  }

  private int vocabularySize = Integer.MAX_VALUE;

  public int getVocabularySize() {
    return vocabularySize;
  }

  /**
   * Sets the size of the vocabulary, that is the number of most
   * frequent words which will be considered when counting word
   * occurences in each document. This is also the size of the
   * resulting vector space.
   */
  public CountVectorizer setVocabularySize(int vocabularySize) {
    this.vocabularySize = vocabularySize;
    return this;
  }

  /**
   * Given a RDD of iterables of strings, return an RDD of word
   * counts. During the process, the algorithm will select the top
   * `vocabularySize` words by frequency. If you have not set the
   * vocabulary size with the `setVocabularySize` method, then the
   * algorithm will use all the words in the dataset.
   *
   * Once the algorithm as built the vacabulary, it proceeds with
   * counting, for each document, the number of occurences of each
   * word of the vocabulary. Each document is then represented by a
   * (sparse) vector of word counts in a space of size
   * `vocabularySize`.
   */
  public <D extends Iterable<String>> JavaRDD<Vector> transform(JavaRDD<D> docs) {
    String[] vocabulary = docs.mapPartitionsToPair((partition) -> {
      Object2IntOpenHashMap<String> wc = new Object2IntOpenHashMap<>();
      wc.defaultReturnValue(0);
      while (partition.hasNext()) {
        Iterable<String> d = partition.next();
        for (String w : d) {
          wc.addTo(w, 1);
        }
      }
      return new IteratorConverter(wc);
    }).reduceByKey((c1, c2) -> c1 + c2)
      .top(vocabularySize, new TupleComparator())
      .stream()
      .map((t) -> t._1())
      .toArray((size) -> new String[size]);

    this.vocabularySize = vocabulary.length;

    Broadcast<String[]> brVocabulary =
      new JavaSparkContext(docs.context()).broadcast(vocabulary);

    JavaRDD<Vector> vectors = docs.map((words) -> {
      String[] mVocab = brVocabulary.getValue();
      Int2IntOpenHashMap wc = new Int2IntOpenHashMap();
      wc.defaultReturnValue(0);
      for (String w : words) {
        for (int i=0; i<mVocab.length; i++) {
          if (w.equals(mVocab[i])) {
            wc.addTo(i, 1);
            break;
          }
        }
      }
      List<Tuple2<Integer, Double>> counts = wc
        .int2IntEntrySet()
        .stream()
        .map((e) -> new Tuple2<>(e.getIntKey(), (double) e.getIntValue()))
        .collect(Collectors.toList());
      return Vectors.sparse(vocabularySize, counts);
    });

    return vectors;
  }

}
