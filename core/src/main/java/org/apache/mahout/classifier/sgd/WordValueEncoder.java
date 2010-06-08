package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Vector;

/**
 * Encodes words as sparse vector updates to a Vector.  Weighting is defined by a
 * sub-class.
 */
public abstract class WordValueEncoder extends RecordValueEncoder {

  public WordValueEncoder(String name) {
    super(name);
    probes = 2;
  }

  /**
   * Adds a value to a vector.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, Vector data) {
    double weight = weight(originalForm);
    for (int i = 0; i < probes; i++) {
      int n = hash(name, originalForm, WORD_LIKE_VALUE_HASH_SEED + i, data.size());
      trace(name, originalForm, n);
      data.set(n, data.get(n) + weight);
    }
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found with
   * associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    return String.format("%s:%s:%.4f", name, originalForm, weight(originalForm));
  }

  protected abstract double weight(String originalForm);
}
