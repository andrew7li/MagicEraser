import styles from "./second.module.scss";

import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import axios from "axios";
import { useEffect, useState } from "react";
import Cropper from "react-easy-crop";
import { ImageSegmentAPIResponse } from "~/types/ISegment";
import { useUploadThing } from "~/utils/uploadthing";
import getCroppedImg from "../../utils/cropImage";
import Resizer from "react-image-file-resizer";

type SecondProps = {
  file: File | null | undefined;
  setWorkflow: (newWorkflow: number) => void;
  setSegmentationData: (response: ImageSegmentAPIResponse) => void;
  setUploadThingUrl: (uploadThingUrl: string) => void;
};

type Area = {
  width: number;
  height: number;
  x: number;
  y: number;
};

export default function Second(props: SecondProps) {
  const { setWorkflow, file, setSegmentationData, setUploadThingUrl } = props;

  const [imageSrc, setImageSrc] = useState("");
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area>();
  const [rotation, _] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // useEffect to handle when the file is set or changed.
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

  /**
   * Function to resize image file before sending it to backend.
   */
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
          // Ensure uri is a base64 string and not null
          if (typeof uri === "string") {
            // Convert base64 to Blob
            const blob = base64ToBlob(uri, "image/jpeg");

            // Convert Blob to File
            const newFile = new File([blob], "resized-image.jpeg", {
              type: "image/jpeg",
            });

            resolve(newFile);
          } else {
            reject(
              new Error("Unexpected URI type from Resizer.imageFileResizer")
            );
          }
        },
        "base64",
        512,
        512
      );
    });

  /**
   * Helper function to convert base64 string to Blob
   */
  function base64ToBlob(base64: string, mimeType: string): Blob {
    const byteCharacters = atob(base64.split(",")[1]);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  }

  const { startUpload, permittedFileInfo } = useUploadThing("imageUploader", {
    onClientUploadComplete: (uploadResponse) => {
      console.log("Uploaded successfully to UploadThing!", uploadResponse);
      console.log("UploadThing's URL", uploadResponse[0].url);
      setUploadThingUrl(uploadResponse[0].url);
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
      .post("https://magic-eraser.abhyudaya.dev/getImageSegments", {
        url: url,
      })
      .then(
        (response) => {
          console.log(response);
          setSegmentationData(response.data);
          setIsUploading(false);

          if (response.data.objects.length === 0) {
            alert(
              "No objects detected! Please upload a new image or change the crop/zoom."
            );
          } else {
            setWorkflow(2);
          }
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
      const croppedImageBlobUrl = await getCroppedImg(
        imageSrc,
        croppedAreaPixels!,
        rotation
      );
      console.log("Successfully cropped!", { croppedImageBlobUrl });

      if (croppedImageBlobUrl === null) {
        throw new Error("Failed to obtain the cropped image.");
      }

      console.log("Successfully cropped!", { croppedImageBlobUrl });

      // Fetch the blob from the blob URL
      const response = await fetch(croppedImageBlobUrl);
      const blob = await response.blob();

      // Create a File from the Blob
      const croppedImageFile = new File([blob], "cropped-image.jpeg", {
        type: "image/jpeg",
      });

      console.log(croppedImageFile);

      // Now resize the image file
      const resizedFile = await resizeFile(croppedImageFile);
      console.log("Successfully resized!", { resizedFile });

      // Proceed with uploading or further processing
      await startUpload([resizedFile]);
    } catch (error) {
      console.error(error);
      setIsUploading(false);
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
          {isUploading && (
            <Box
              sx={{
                borderRadius: "1rem",
                position: "absolute",
                top: "0",
                left: "0",
                width: "100%",
                height: "100%",
                backgroundColor: "rgba(35, 1, 45, 0.8)",
                zIndex: "1000",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <CircularProgress sx={{ color: "#D04D76" }} />
            </Box>
          )}
        </div>
      </div>
    </>
  );
}
