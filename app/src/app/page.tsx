"use client";
import styles from "./test.module.scss";

import { UploadDropzone } from "@uploadthing/react";
import { useUploadThing } from "./../utils/uploadthing";
import TopNav from "./TopNav";

async function compress(file: File) {
  // Run some compression algorithm on the file
  return file;
}

export default function Home() {
  const { startUpload } = useUploadThing("profileImage", {
    onClientUploadComplete: () => {
      alert("Upload Completed");
    },
  });

  return (
    <div className={styles.body}>
      <TopNav />
      <div className={styles.headerContainer}>
        <p className={styles.headerText}>
          remove objects with <span>magic eraser</span>
        </p>
      </div>
      <div className={styles.imageContainer}>
        <div className={styles.imageDemoContainer}>
          <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />
        </div>
        <div className={styles.imageUploadContainer}>
          <input
            type="file"
            id="inputFile"
            style={{ display: "none" }} // Hide the input but keep it functional
            accept="image/jpeg, image/png, image/webp, image/bmp" // Specify accepted file types
            multiple // Allow multiple file uploads if necessary
            // onChange={handleFileUpload} // Implement a function to handle file uploads
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;

              // Do something with the file before uploading
              const compressed = await compress(file);

              // Then start the upload of the compressed file
              await startUpload([compressed]);
            }}
          />
          <div className={styles.uploadImageDragAndDropContainer}>
            {/* <label htmlFor="inputFile" className={styles.uploadImageTextLabel}>
              <span>
                <MdOutlineFileUpload size={20} />
              </span>
              <ul>
                <li>Drag or drop images here</li>
                <li>JPG, PNG, JPEG, WEBP, BMP· Max size 5Mb</li>
              </ul>
            </label>

            <p>
              *All files are stored privately & encrypted. Only you will see
              them.
            </p> */}
            <UploadDropzone
              endpoint="imageUploader"
              appearance={{
                button: {
                  background: "rgba(129, 0, 165, 0.2)",
                  margin: "auto",
                  borderRadius: "1rem",
                  height: "3.5rem",
                  border: "none",
                  color: "#000000",
                  fontSize: "20px",
                  letterSpacing: "0.5px",
                  fontWeight: "700",
                  display: "flex",
                  position: "relative",
                  justifyContent: "center",
                  alignItems: "center",
                  width: "90%",
                  padding: "10px",
                  textTransform: "lowercase",
                },
                container: {
                  width: "90%",
                  cursor: "pointer",
                },
              }}
              onClientUploadComplete={(res) => {
                // Do something with the response
                console.log("Files: ", res);
                alert("Upload Completed");
              }}
              onUploadError={(error: Error) => {
                // Do something with the error.
                alert(`ERROR! ${error.message}`);
              }}
            />
          </div>

          <label htmlFor="inputFile" className={styles.uploadImageButtonLabel}>
            {/* <div className={styles.uploadButton}>Upload</div> */}
            {/* <UploadButton
              endpoint="imageUploader"
              onClientUploadComplete={(res) => {
                // Do something with the response
                console.log("Files: ", res);
                alert("Upload Completed");
              }}
              onUploadError={(error: Error) => {
                // Do something with the error.
                alert(`ERROR! ${error.message}`);
              }}
            /> */}
          </label>

          <div className={styles.sampleImagesContainer}>
            <p>Try one of these now!</p>
            <ul>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
              <li>
                <img src="/IMG_8916.jpeg" alt="Demo of Magic Eraser" />{" "}
              </li>
            </ul>
          </div>
        </div>
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

// "use client";

// import { UploadButton } from "./../utils/uploadthing.ts";

// export default function Home() {
//   return (
//     <main className="flex min-h-screen flex-col items-center justify-between p-24">
// <UploadButton
//   endpoint="imageUploader"
//   onClientUploadComplete={(res) => {
//     // Do something with the response
//     console.log("Files: ", res);
//     alert("Upload Completed");
//   }}
//   onUploadError={(error: Error) => {
//     // Do something with the error.
//     alert(`ERROR! ${error.message}`);
//   }}
// />
//     </main>
//   );
// }
