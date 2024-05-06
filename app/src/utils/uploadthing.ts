import {
    generateUploadButton,
    generateUploadDropzone,
  } from "@uploadthing/react";
   
  import type { OurFileRouter } from "~/app/api/uploadthing/core";
  import { generateSolidHelpers } from "@uploadthing/solid";
 
// import type { OurFileRouter } from "~/server/uploadthing";
   
  export const UploadButton = generateUploadButton<OurFileRouter>();
  export const UploadDropzone = generateUploadDropzone<OurFileRouter>();
  export const { useUploadThing } = generateSolidHelpers<OurFileRouter>();