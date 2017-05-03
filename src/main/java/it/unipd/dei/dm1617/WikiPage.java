package it.unipd.dei.dm1617;

import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Class that represents a single Wikipedia page. Modify it at your
 * liking, but bear in mind that to support automatic conversion to
 * and from JSON file datasets this class must be a Java Bean: the
 * fields are private (and in our case have the same name as the
 * fields of the JSON objects we read from files) and for each field
 * there shuold be a pair of `getField` and `setField` methods.
 */
public class WikiPage implements Serializable {

  public static Encoder<WikiPage> getEncoder() {
    return Encoders.bean(WikiPage.class);
  }

  private long id;

  private String title;

  private String text;

  private String[] categories;

  public WikiPage() { }

  public WikiPage(long id, String title, String text, String[] categories) {
    this.id = id;
    this.title = title;
    this.text = text;
    this.categories = categories;
  }

  public long getId() {
    return id;
  }

  public void setId(long id) {
    this.id = id;
  }

  public String getTitle() {
    return title;
  }

  public void setTitle(String title) {
    this.title = title;
  }

  public String getText() {
    return text;
  }

  public void setText(String text) {
    this.text = text;
  }

  public String[] getCategories() {
    return categories;
  }

  public void setCategories(String[] categories) {
    this.categories = categories;
  }

  @Override
  public String toString() {
    return "(" + id + ") `" + title + "` " + Arrays.asList(categories) + " " + text;
  }
}
