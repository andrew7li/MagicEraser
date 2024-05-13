import styles from "./third.module.scss";
import Box from "@mui/material/Box";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import { useState } from "react";
import { ImageSegmentAPIResponse } from "~/types/ISegment";

const mockData = {
  objects: [
    {
      uuid: "a6c7a62c-7618-4c1d-a6b8-13d984ddc142",
      objectType: "bicycle",
      topLeft: {
        x: 1500,
        y: 681,
      },
      topRight: {
        x: 1585,
        y: 681,
      },
      bottomRight: {
        x: 1585,
        y: 681,
      },
      bottomLeft: {
        x: 1500,
        y: 681,
      },
    },
    {
      uuid: "043f9838-564e-4495-881b-d1fca9a8fefb",
      objectType: "bicycle",
      topLeft: {
        x: 1380,
        y: 681,
      },
      topRight: {
        x: 1442,
        y: 681,
      },
      bottomRight: {
        x: 1442,
        y: 681,
      },
      bottomLeft: {
        x: 1380,
        y: 681,
      },
    },
    {
      uuid: "fa55792b-eb47-44a7-a871-be27fc6d83b5",
      objectType: "potted plant",
      topLeft: {
        x: 1540,
        y: 791,
      },
      topRight: {
        x: 1597,
        y: 791,
      },
      bottomRight: {
        x: 1597,
        y: 791,
      },
      bottomLeft: {
        x: 1540,
        y: 791,
      },
    },
    {
      uuid: "720007ba-557a-474e-ab8e-44a09e2a5917",
      objectType: "bicycle",
      topLeft: {
        x: 390,
        y: 673,
      },
      topRight: {
        x: 505,
        y: 673,
      },
      bottomRight: {
        x: 505,
        y: 673,
      },
      bottomLeft: {
        x: 390,
        y: 673,
      },
    },
  ],
};

type ThirdProps = {
  segmentationData: ImageSegmentAPIResponse | undefined;
  uploadThingUrl: string;
};

export default function Third(props: ThirdProps) {
  const { segmentationData, uploadThingUrl } = props;
  const [object, setObject] = useState("");

  const handleChange = (event: SelectChangeEvent) => {
    setObject(event.target.value as string);
  };

  return !segmentationData ? (
    <div>Error! No data found!</div>
  ) : (
    <>
      <div className={styles.leftContainer}>
        <img src={uploadThingUrl} />
      </div>
      <div className={styles.rightContainer}>
        <Box sx={{ minWidth: 120 }}>
          <FormControl fullWidth>
            <InputLabel id="select-label">-select-</InputLabel>
            <Select
              labelId="select-label"
              id="simple-select"
              value={object}
              label="Object"
              onChange={handleChange}
              MenuProps={{
                PaperProps: {
                  sx: {
                    bgcolor: "rgb(252, 243, 255)",
                    width: 300,
                    // border: "6px solid rgba(129, 0, 165, 0.3)",
                    border: "4px solid rgba(91, 0, 119, 0.425)",
                    borderRadius: "15px",
                    "& .MuiMenuItem-root": {
                      padding: 2,
                    },
                  },
                },
              }}
            >
              {segmentationData.objects.map((element) => (
                <MenuItem value={element.uuid}>{element.objectType}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        {/* <div>Image Segments</div>
        {mockData.objects.map((element) => (
          <div key={element.uuid}>{element.objectType}</div>
        ))} */}
      </div>
    </>
  );
}
