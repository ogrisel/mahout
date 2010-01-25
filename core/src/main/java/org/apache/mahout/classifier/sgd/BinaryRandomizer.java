package org.apache.mahout.classifier.sgd;

import java.util.List;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Multiplies a sparse vector in the form of a term list by a random binary matrix.
 */
public class BinaryRandomizer extends TermRandomizer {
  private final int probes;
  private final int numFeatures;

  public BinaryRandomizer(int probes, int numFeatures) {
    this.probes = probes;
    this.numFeatures = numFeatures;
  }

  @Override
  public Vector randomizedInstance(List<String> terms, int window, boolean allPairs) {
    Vector instance = new RandomAccessSparseVector(getNumFeatures(), Math.min(terms.size() * getProbes(), 20));
    int n = 0;
    for (String term : terms) {
      for (int probe = 0; probe < getProbes(); probe++) {
        int i = hash(term, probe, getNumFeatures());
        instance.setQuick(i, instance.getQuick(i) + 1);
      }

      if (allPairs) {
        for (String other : terms) {
          for (int probe = 0; probe < getProbes(); probe++) {
            int i = hash(term, other, probe, getNumFeatures());
            instance.setQuick(i, instance.getQuick(i) + 1);
          }
        }
      }

      if (window > 0) {
        for (int j = Math.max(0, n - window); j < n; j++) {
          for (int probe = 0; probe < getProbes(); probe++) {
            int i = hash(term, terms.get(j), probe, getNumFeatures());
            instance.setQuick(i, instance.getQuick(i) + 1);
          }
        }
      }
      n++;
    }
    return instance;
  }

  public int getProbes() {
    return probes;
  }

  public int getNumFeatures() {
    return numFeatures;
  }
}
