import styles from "./second.module.scss";

import Button from "@mui/material/Button";
import axios from "axios";
import { useEffect, useState } from "react";
import Cropper from "react-easy-crop";
import getCroppedImg from "../../utils/cropImage";

import { useUploadThing } from "~/utils/uploadthing";

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

  const [imageSrc, setImageSrc] = useState("");
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area>();
  const [croppedImage, setCroppedImage] = useState<string | null>();
  const [rotation, setRotation] = useState(0);
  const [showCroppedImage, setShowCroppedImage] = useState(false);

  // Function to handle when the file is set or changed
  useEffect(() => {
    if (file) {
      const fileUrl = URL.createObjectURL(file);
      setImageSrc(fileUrl);

      // Cleanup the URL when the component unmounts or file changes
      return () => {
        URL.revokeObjectURL(fileUrl);
      };
    }
  }, [file]);

  const { startUpload, permittedFileInfo } = useUploadThing("imageUploader", {
    onClientUploadComplete: () => {
      // alert("uploaded successfully!");
    },
    onUploadError: () => {
      // alert("error occurred while uploading");
    },
    onUploadBegin: () => {
      // alert("upload has begun");
    },
  });

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

  const onCropComplete = (croppedArea: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels);
  };

  const handleGetSegmentButtonClick = async () => {
    // setShowCroppedImage(true);
    try {
      const croppedImage = await getCroppedImg(
        imageSrc,
        croppedAreaPixels!,
        rotation
      );
      console.log("Successfully cropped!", { croppedImage });
      setCroppedImage(croppedImage);

      // Fetch the blob from the local blob URL
      fetch(croppedImage!)
        .then((response) => response.blob())
        .then((blob) => {
          // Now that you have the blob, create a File object
          const file = new File([blob], "cropped-image.jpeg", {
            type: "image/jpeg",
          });

          // Upload the file
          startUpload([file]);
        })
        .catch((error) =>
          console.error("Failed to fetch blob from URL:", error)
        );

      // callImageSegmentsAPI(croppedImage!);
    } catch (e) {
      console.error(e);
    }
  };

  return !file ? (
    <div>Error! No file found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>
        <Cropper
          image={imageSrc}
          crop={crop}
          zoom={zoom}
          aspect={1}
          onCropChange={setCrop}
          onZoomChange={setZoom}
          onCropComplete={onCropComplete}
        />
      </div>
      <div className={styles.rightContainer}>
        <p>
          Our application only supports square images. Please use the crop
          feature to ensure proper square dimensions for the image!
        </p>
        <div
          className={styles.getSegmentButton}
          onClick={handleGetSegmentButtonClick}
        >
          Get Image Segments
        </div>
      </div>
    </>
  );
}
