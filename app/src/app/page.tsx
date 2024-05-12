"use client";
import styles from "./test.module.scss";

import TopNav from "./TopNav";

import axios from "axios";
import Resizer from "react-image-file-resizer";

import { useState } from "react";
import First from "./workflows/first";
import Second from "./workflows/second";
import Third from "./workflows/third";

export default function Home() {
  const [file, setFile] = useState<File | null>();
  const [workflow, setWorkflow] = useState(0);

  const callImageSegmentsAPI = (url: string) => {
    axios
      .post("http://localhost:8000/getImageSegments", {
        url: url,
      })
      .then(
        (response) => {
          console.log(response);
        },
        (error) => {
          console.log(error);
        }
      );
  };

  const resizeFile = (file: File): Promise<File> =>
    new Promise((resolve, reject) => {
      Resizer.imageFileResizer(
        file,
        512,
        512,
        "JPEG",
        100,
        0,
        (uri) => {
          if (typeof uri === "string") {
            // Convert base64 string to Blob
            const blob = base64ToBlob(uri, "image/jpeg");
            // Create a new File from the Blob
            const newFile = new File([blob], "resized-" + file.name, {
              type: "image/jpeg",
            });
            resolve(newFile);
          } else {
            reject(new Error("URI is not in the expected string format"));
          }
        },
        "base64",
        512,
        512
      );
    });

  /**
   * Helper function to convert base64 to Blob
   */
  function base64ToBlob(base64: string, mimeType: string) {
    const byteString = atob(base64.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeType });
  }

  return (
    <div className={styles.body}>
      <TopNav setWorkflow={setWorkflow} setFile={setFile} />
      <div className={styles.headerContainer}>
        <p className={styles.headerText}>
          remove objects with <span>magic eraser</span>
        </p>
      </div>

      <div className={styles.middleContainer}>
        {workflow === 0 ? (
          <First setWorkflow={setWorkflow} setFile={setFile} />
        ) : workflow === 1 ? (
          <Second />
        ) : workflow === 2 ? (
          <Third />
        ) : null}
      </div>

      <div className={styles.descriptionContainer}>
        <strong>Instantly modify your images to create a visual impact</strong>
        <br />
        <p>
          Experience the magic of removing objects beyond what is traditionally
          capable
        </p>
      </div>
    </div>
  );
}
