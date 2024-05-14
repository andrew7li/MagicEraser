import axios from "axios";
import { ChangeEvent, useEffect, useRef, useState } from "react";

import styles from "./third.module.scss";

import { ImageSegmentAPIResponse } from "~/types/ISegment";
import Canvas from "../components/canvas";

import { TextField, ThemeProvider, createTheme } from "@mui/material";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";

type ThirdProps = {
  segmentationData: ImageSegmentAPIResponse | undefined;
  uploadThingUrl: string | undefined;
  setWorkflow: (newWorkflow: number) => void;
  setFinalOutputUrl: (finalOutputUrl: string) => void;
};

/**
 * MUI Theme to be consistent with the app's primary purple color.
 */
const theme = createTheme({
  palette: {
    primary: {
      main: "rgba(129, 0, 165, 0.4)",
    },
  },
});

export default function Third(props: ThirdProps) {
  const { segmentationData, uploadThingUrl, setWorkflow, setFinalOutputUrl } =
    props;
  const [objectIdx, setObjectIdx] = useState<string>("0");
  const [uuid, setUuid] = useState<string>(
    segmentationData?.objects[Number(objectIdx)].uuid ?? ""
  );
  const [prompt, setPrompt] = useState<string>("");
  const [isInpainting, setIsInpainting] = useState(false);

  /**
   * Handler function for when the prompt changes.
   */
  const handlePromptChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPrompt(event.target.value);
  };

  /**
   * Handler function for when the segment object changes.
   */
  const handleSegmentObjectChange = (event: SelectChangeEvent) => {
    const idx = event.target.value;
    setObjectIdx(idx);
    setUuid(segmentationData!.objects[Number(idx)].uuid as string);
  };

  /**
   * Handler function for getting the output when the output button is clicked.
   */
  const handleGetOutputButtonClick = async () => {
    if (!prompt || !uuid) {
      alert("Prompt or UUID cannot be empty!");
      return;
    }
    setIsInpainting(true);
    axios
      .post("https://magic-eraser.abhyudaya.dev/inpaintImage", {
        url: uploadThingUrl,
        uuid: uuid,
        prompt: prompt,
      })
      .then(
        (response) => {
          setFinalOutputUrl(response.data.url);
          setWorkflow(3);
          setIsInpainting(false);
        },
        (error) => {
          console.error(error);
          setIsInpainting(false);
        }
      );
  };

  return !segmentationData || !uploadThingUrl ? (
    <div>Error! No data found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>
        <Canvas
          segmentationObject={segmentationData.objects[Number(objectIdx)]}
          uploadThingUrl={uploadThingUrl}
        />
      </div>
      <div className={styles.rightContainer}>
        <div
          style={{
            width: "90%",
          }}
        >
          <ThemeProvider theme={theme}>
            <FormControl
              fullWidth
              sx={{
                "& .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline": {
                  borderColor: "rgba(129, 0, 165, 0.8) !important", // Normal state border color
                },
                "&:hover .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline":
                  {
                    borderColor: "rgba(129, 0, 165, 0.4) !important", // Hover state border color
                  },
                "&.Mui-focused .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline":
                  {
                    borderColor: "rgba(129, 0, 165, 0.5) !important", // Focus state border color
                  },
              }}
            >
              <InputLabel id="select-label">Segment</InputLabel>
              <Select
                labelId="select-label"
                id="simple-select"
                value={String(objectIdx)}
                label="Segment"
                onChange={handleSegmentObjectChange}
                sx={{
                  textAlign: "left",
                  width: "100%",
                }}
                MenuProps={{
                  PaperProps: {
                    sx: {
                      bgcolor: "rgb(252, 243, 255)",
                      border: "4px solid rgba(91, 0, 119, 0.425)",
                      borderRadius: "10px",
                      "& .MuiMenuItem-root": {
                        padding: "10px 15px",
                      },
                    },
                  },
                }}
              >
                {segmentationData.objects.map((element, idx) => (
                  <MenuItem key={element.uuid} value={idx}>
                    {element.objectType}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </ThemeProvider>
        </div>
        <div
          style={{
            width: "100%",
          }}
        >
          <ThemeProvider theme={theme}>
            <TextField
              id="outlined-textarea"
              label="Prompt Magic Eraser"
              placeholder="remove this object from the image and inpaint it naturally..."
              multiline
              rows={4}
              sx={{
                width: "90%",
                "& .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline": {
                  borderColor: "rgba(129, 0, 165, 0.8)", // Normal state border color
                },
                "& .MuiOutlinedInput-root:hover .MuiOutlinedInput-notchedOutline":
                  {
                    borderColor: "rgba(129, 0, 165, 0.4)", // Hover state border color
                  },
                "& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline":
                  {
                    borderColor: "rgba(129, 0, 165, 0.5)", // Focus state border color
                  },
              }}
              value={prompt}
              onChange={handlePromptChange}
            />
          </ThemeProvider>
        </div>
        <div
          className={styles.getOutputButton}
          onClick={handleGetOutputButtonClick}
        >
          Get output
          {isInpainting && (
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
