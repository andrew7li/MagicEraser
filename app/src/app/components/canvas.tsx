import { useEffect, useRef } from "react";
import { SegmentObject } from "~/types/ISegment";

type CanvasProps = {
  segmentationObject: SegmentObject;
  uploadThingUrl: string | undefined;
};

export default function Canvas(props: CanvasProps) {
  const { segmentationObject, uploadThingUrl } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!uploadThingUrl || !canvasRef.current || !segmentationObject) {
      return;
    }
    const canvas = canvasRef.current as HTMLCanvasElement;
    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    // Load and draw the background image
    const backgroundImage = new Image();
    backgroundImage.src = uploadThingUrl;
    backgroundImage.onload = () => {
      context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

      // Draw rectangles after the image has loaded
      drawRectangles(context);
    };
  }, [segmentationObject, uploadThingUrl]);

  /**
   * Function to draw rectangles.
   */
  const drawRectangles = (context: CanvasRenderingContext2D) => {
    const rectangles = [
      {
        x: segmentationObject.topLeft.x,
        y: segmentationObject.topLeft.y,
        width: segmentationObject.topRight.x - segmentationObject.topLeft.x,
        height: segmentationObject.bottomLeft.y - segmentationObject.topLeft.y,
      },
    ];

    rectangles.forEach((rect) => {
      context.beginPath();
      context.rect(rect.x, rect.y, rect.width, rect.height);
      context.fillStyle = "rgba(255, 0, 0, 0.5)"; // Semi-transparent red
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = "red";
      context.stroke();
    });
  };

  return (
    <>
      <canvas
        ref={canvasRef}
        width={512}
        height={512}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
        }}
      ></canvas>
    </>
  );
}
