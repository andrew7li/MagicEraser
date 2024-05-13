import styles from "./second.module.scss";
import Cropper from "react-easy-crop";
import { useState } from "react";
import getCroppedImg from "../../utils/cropImage";
import Button from "@mui/material/Button";

import axios from "axios";

type SecondProps = {
  setWorkflow: (newWorkflow: number) => void;
  file: File | null | undefined | null | undefined;
};

type Area = {
  width: number;
  height: number;
  x: number;
  y: number | null | undefined;
};

export default function Second(props: SecondProps) {
  const { setWorkflow, file } = props;

  const callImageSegmentsAPI = (url: string) => {
    axios
      .post(
        "https://fur-mitsubishi-guru-generated.trycloudflare.com/getImageSegments",
        {
          url: url,
        }
      )
      .then(
        (response) => {
          console.log(response);
        },
        (error) => {
          console.log(error);
        }
      );
  };

  // const handleGetSegmentButtonClick = () => {
  //   callImageSegmentsAPI(file);
  // };

  return !file ? (
    <div>Error! No file found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>Fill me in!</div>
      <div className={styles.rightContainer}>
        <p>
          Our application only supports square images. Please use the crop
          feature to ensure proper square dimensions for the image!
        </p>
        <div
          className={styles.getSegmentButton}
          // onClick={handleGetSegmentButtonClick}
        >
          Get Image Segments
        </div>
      </div>
    </>
  );
}
