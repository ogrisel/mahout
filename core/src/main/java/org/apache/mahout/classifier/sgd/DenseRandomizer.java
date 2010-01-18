package org.apache.mahout.classifier.sgd;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.Random;

/**
 * Implements multiplication by a random matrix.
 */
public class DenseRandomizer extends TermRandomizer {
  private Random gen = RandomUtils.getRandom();
  private int numFeatures;

  public DenseRandomizer(int numFeatures) {
    this.numFeatures = numFeatures;
  }

  public Vector randomizedInstance(List<String> terms, int window, boolean allPairs) {
    Vector instance = new DenseVector(getNumFeatures());
    for (int i = 0; i < getNumFeatures(); i++) {
      double v = 0;
      for (String term : terms) {
        gen.setSeed(hash(term, i, getNumFeatures()));
        v = gen.nextGaussian();
      }

      if (allPairs) {
        for (String term : terms) {
          for (String other : terms) {
            gen.setSeed(hash(term, other, i, getNumFeatures()));
            v += gen.nextGaussian();
          }
        }
      }

      if (window > 0) {
        for (int n = 0; n < terms.size(); n++) {
          for (int m = Math.max(0, n - window); m < Math.min(terms.size(), n + window); m++) {
            if (n != m) {
              gen.setSeed(hash(terms.get(n), terms.get(m), i, getNumFeatures()));
              v += gen.nextGaussian();
            }
          }
        }
      }

      instance.setQuick(i, v);
    }

    return instance;
  }

  public int getNumFeatures() {
    return numFeatures;
  }
}
