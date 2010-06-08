package org.apache.mahout.classifier.sgd;

import com.google.common.collect.ImmutableMap;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CsvRecordFactoryTest {
  @Test
  public void testAddToVector() {
    CsvRecordFactory csv = new CsvRecordFactory(2);
    csv.firstLine("y", ImmutableMap.of("x1", "n", "x2", "w", "x3", "t"), "z,x1,y,x2,x3,q");

    Vector v = new DenseVector(2000);
    int t = csv.processLine("ignore,3.1,yes,tiger, \"this is text\",ignore", v);
    assertEquals(0, t);
    // should have 9 values set
    assertEquals(9.0, v.norm(0), 0);
    // all should be = 1 except for the 3.1
    assertEquals(3.1, v.maxValue(), 0);
    v.set(v.maxValueIndex(), 0);
    assertEquals(8.0, v.norm(0), 0);
    assertEquals(8.0, v.norm(1), 0);
    assertEquals(1.0, v.maxValue(), 0);

    v.assign(0);
    t = csv.processLine("ignore,5.3,no,line, \"and more text and more\",ignore", v);
    assertEquals(1, t);

    // should have 9 values set
    assertEquals(9.0, v.norm(0), 0);
    // all should be = 1 except for the 3.1
    assertEquals(5.3, v.maxValue(), 0);
    v.set(v.maxValueIndex(), 0);
    assertEquals(8.0, v.norm(0), 0);
    assertEquals(12.0, v.norm(1), 0);
    assertEquals(2, v.maxValue(), 0);

    v.assign(0);
    t = csv.processLine("ignore,5.3,invalid,line, \"and more text and more\",ignore", v);
    assertEquals(1, t);

    // should have 9 values set
    assertEquals(9.0, v.norm(0), 0);
    // all should be = 1 except for the 3.1
    assertEquals(5.3, v.maxValue(), 0);
    v.set(v.maxValueIndex(), 0);
    assertEquals(8.0, v.norm(0), 0);
    assertEquals(12.0, v.norm(1), 0);
    assertEquals(2, v.maxValue(), 0);
  }
}
