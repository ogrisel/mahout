package org.apache.mahout.classifier.sgd;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class WordLikeValueEncoderTest {
  @Test
  public void testAddToVector() {
    RecordValueEncoder enc = new StaticWordValueEncoder("word");
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);
    enc.addToVector("word2", v);
    Iterator<Vector.Element> i = v.iterateNonZero();
    Iterator<Integer> j = ImmutableList.of(7, 118, 119, 199).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
      assertEquals(1, element.get(), 0);
    }
    assertFalse(j.hasNext());
  }

  @Test
  public void testAsString() {
    RecordValueEncoder enc = new StaticWordValueEncoder("word");
    assertEquals("word:w1:1.0000", enc.asString("w1"));
  }

  @Test
  public void testStaticWeights() {
    StaticWordValueEncoder enc = new StaticWordValueEncoder("word");
    enc.setDictionary(ImmutableMap.<String, Double>of("word1", 3.0, "word2", 1.5));
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);
    enc.addToVector("word2", v);
    enc.addToVector("word3", v);
    Iterator<Vector.Element> i = v.iterateNonZero();
    Iterator<Integer> j = ImmutableList.of(7, 101, 118, 119, 152, 199).iterator();
    Iterator<Double> k = ImmutableList.of(3.0, 0.75, 1.5, 1.5, 0.75, 3.0).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
      assertEquals(k.next(), element.get(), 0);
    }
    assertFalse(j.hasNext());
  }

  @Test
  public void testDynamicWeights() {
    RecordValueEncoder enc = new AdaptiveWordValueEncoder("word");
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);  // weight is log(2/1.5)
    enc.addToVector("word2", v);  // weight is log(3.5 / 1.5)
    enc.addToVector("word1", v);  // weight is log(4.5 / 2.5) (but overlays on first value)
    enc.addToVector("word3", v);  // weight is log(6 / 1.5)
    Iterator<Vector.Element> i = v.iterateNonZero();
    Iterator<Integer> j = ImmutableList.of(7, 101, 118, 119, 152, 199).iterator();
    Iterator<Double> k = ImmutableList.of(Math.log(2 / 1.5) + Math.log(4.5 / 2.5), Math.log(6 / 1.5), Math.log(3.5 / 1.5), Math.log(3.5 / 1.5), Math.log(6 / 1.5), Math.log(2 / 1.5) + Math.log(4.5 / 2.5)).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
      assertEquals(k.next(), element.get(), 1e-6);
    }
    assertFalse(j.hasNext());
  }
}
