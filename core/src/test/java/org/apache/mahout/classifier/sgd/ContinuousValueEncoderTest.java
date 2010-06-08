package org.apache.mahout.classifier.sgd;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class ContinuousValueEncoderTest {
  @Test
  public void testAddToVector() {
    RecordValueEncoder enc = new ContinuousValueEncoder("foo");
    Vector v1 = new DenseVector(20);
    enc.addToVector("123", v1);
    assertEquals(123, v1.maxValue(), 0);
    assertEquals(123, v1.norm(1), 0);

    Vector v2 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("123", v2);
    assertEquals(123, v2.maxValue(), 0);
    assertEquals(2 * 123, v2.norm(1), 0);

    v1 = v2.minus(v1);
    assertEquals(123, v1.maxValue(), 0);
    assertEquals(123, v1.norm(1), 0);

    Vector v3 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("100", v3);
    v1 = v2.minus(v3);
    assertEquals(23, v1.maxValue(), 0);
    assertEquals(2 * 23, v1.norm(1), 0);

    enc.addToVector("7", v1);
    assertEquals(30, v1.maxValue(), 0);
    assertEquals(2 * 30, v1.norm(1), 0);
    assertEquals(0, v1.get(5), 0);
    assertEquals(30, v1.get(18), 0);

    try {
      enc.addToVector("foobar", v1);
      fail("Should have noticed back numeric format");
    } catch (NumberFormatException e) {
      assertEquals("For input string: \"foobar\"", e.getMessage());
    }
  }

  @Test
  public void testAsString() {
    ContinuousValueEncoder enc = new ContinuousValueEncoder("foo");
    assertEquals("foo:123", enc.asString("123"));
  }

}
