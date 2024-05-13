import styles from "./second.module.scss";
import Cropper from "react-easy-crop";
import { useState } from "react";
import getCroppedImg from "../../utils/cropImage";
import Button from "@mui/material/Button";

import axios from "axios";

type SecondProps = {
  setWorkflow: (newWorkflow: number) => void;
  file: File | null | undefined;
};

type Area = {
  width: number;
  height: number;
  x: number;
  y: number;
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

  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area>();
  const [croppedImage, setCroppedImage] = useState<string | null>();
  const [rotation, setRotation] = useState(0);
  const [showCroppedImage, setShowCroppedImage] = useState(false);

  const img =
    "https://img.huffingtonpost.com/asset/5ab4d4ac2000007d06eb2c56.jpeg?cache=sih0jwle4e&ops=1910_1000";

  const onCropComplete = (croppedArea: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels);
  };

  const handleButtonClick = async () => {
    setShowCroppedImage(true);
    try {
      const croppedImage = await getCroppedImg(
        img,
        croppedAreaPixels!,
        rotation
      );
      console.log("done", { croppedImage });
      setCroppedImage(croppedImage);
    } catch (e) {
      console.error(e);
    }
  };

  // const handleGetSegmentButtonClick = () => {
  //   callImageSegmentsAPI(file);
  // };

  return !file ? (
    <div>Error! No file found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>
        <Cropper
          image={img}
          crop={crop}
          zoom={zoom}
          aspect={1}
          onCropChange={setCrop}
          onZoomChange={setZoom}
          onCropComplete={onCropComplete}
        />
        <Button onClick={handleButtonClick} variant="contained" color="primary">
          Show Result
        </Button>
      </div>
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
