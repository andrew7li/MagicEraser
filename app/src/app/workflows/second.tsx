import styles from "./second.module.scss";

import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import axios from "axios";
import { useEffect, useState } from "react";
import Cropper from "react-easy-crop";
import { ImageSegmentAPIResponse } from "~/types/ISegment";
import { useUploadThing } from "~/utils/uploadthing";
import getCroppedImg from "../../utils/cropImage";

type SecondProps = {
  setWorkflow: (newWorkflow: number) => void;
  file: File | null | undefined;
  setSegmentationData: (response: ImageSegmentAPIResponse) => void;
};

type Area = {
  width: number;
  height: number;
  x: number;
  y: number;
};

export default function Second(props: SecondProps) {
  const { setWorkflow, file, setSegmentationData } = props;

  const [imageSrc, setImageSrc] = useState("");
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area>();
  const [rotation, _] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

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
    onClientUploadComplete: (uploadResponse) => {
      console.log("Uploaded successfully to UploadThing!", uploadResponse);
      console.log("UploadThing's URL", uploadResponse[0].url);
      callImageSegmentsAPI(uploadResponse[0].url);
    },
    onUploadError: () => {
      alert("error occurred while uploading");
    },
    onUploadBegin: () => {
      // NOTE: See if backend needs the image to be 512 x 512
    },
  });

  const callImageSegmentsAPI = (url: string) => {
    axios
      .post(
        "https://floating-likely-dover-windows.trycloudflare.com/getImageSegments",
        {
          url: url,
        }
      )
      .then(
        (response) => {
          console.log(response);
          setSegmentationData(response.data);
          setIsUploading(false);
          setWorkflow(2);
        },
        (error) => {
          console.log(error);
          setIsUploading(false);
        }
      );
  };

  const onCropComplete = (croppedArea: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels);
  };

  const handleGetSegmentButtonClick = async () => {
    setIsUploading(true);
    try {
      const croppedImage = await getCroppedImg(
        imageSrc,
        croppedAreaPixels!,
        rotation
      );
      console.log("Successfully cropped!", { croppedImage });

      // Fetch the blob from the local blob URL
      fetch(croppedImage!)
        .then((response) => response.blob())
        .then(async (blob) => {
          // Now that you have the blob, create a File object
          const file = new File([blob], "cropped-image.jpeg", {
            type: "image/jpeg",
          });

          // Upload the file
          await startUpload([file]);
        })
        .catch((error) => {
          console.error("Failed to fetch blob from URL:", error);
          setIsUploading(false);
        });
    } catch (e) {
      console.error(e);
      setIsUploading(false);
    }
  };

  return !file ? (
    <div>Error! No file found!</div>
  ) : (
    <>
      {isUploading && (
        <Box
          sx={{
            position: "absolute",
            top: "0",
            left: "0",
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0, 0, 0, 0.4)",
            zIndex: "1000",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <CircularProgress />
        </Box>
      )}
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
