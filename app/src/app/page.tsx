"use client";
import styles from "./test.module.scss";

import TopNav from "./TopNav";

import { useState } from "react";
import First from "./workflows/first";
import Second from "./workflows/second";
import Third from "./workflows/third";

import { IoArrowBackOutline } from "react-icons/io5";
import { ImageSegmentAPIResponse } from "~/types/ISegment";
import Fourth from "./workflows/fourth";

export default function Home() {
  const [file, setFile] = useState<File | null>();
  const [workflow, setWorkflow] = useState(0);
  const [segmentationData, setSegmentationData] =
    useState<ImageSegmentAPIResponse>();
  const [uploadThingUrl, setUploadThingUrl] = useState<string>();
  const [finalOutputUrl, setFinalOutputUrl] = useState<string>();

  return (
    <div className={styles.body}>
      <TopNav setWorkflow={setWorkflow} setFile={setFile} />
      <div className={styles.headerContainer}>
        <p className={styles.headerText}>
          remove objects with <span>magic eraser</span>
        </p>
      </div>

      <div className={styles.middleContainer}>
        {workflow !== 0 && (
          <IoArrowBackOutline
            onClick={() => {
              setWorkflow(workflow - 1);
            }}
            className={styles.backArrow}
            size={30}
          />
        )}
        {workflow === 0 ? (
          <First setWorkflow={setWorkflow} setFile={setFile} />
        ) : workflow === 1 ? (
          <Second
            file={file}
            setWorkflow={setWorkflow}
            setSegmentationData={setSegmentationData}
            setUploadThingUrl={setUploadThingUrl}
          />
        ) : workflow === 2 ? (
          <Third
            segmentationData={segmentationData}
            uploadThingUrl={uploadThingUrl}
            setWorkflow={setWorkflow}
            setFinalOutputUrl={setFinalOutputUrl}
          />
        ) : workflow === 3 ? (
          <Fourth finalOutputUrl={finalOutputUrl} setWorkflow={setWorkflow} />
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
