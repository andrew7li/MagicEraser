export type ImageSegmentAPIResponse = {
  objects: SegmentObject[];
};

type SegmentObject = {
  uuid: string;
  objectType: string;
  topLeft: {
    x: number;
    y: number;
  };
  topRight: {
    x: number;
    y: number;
  };
  bottomRight: {
    x: number;
    y: number;
  };
  bottomLeft: {
    x: number;
    y: number;
  };
  confidence: number;
};
