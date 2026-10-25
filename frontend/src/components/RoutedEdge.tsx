import { BaseEdge, getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { createContext, memo, useContext, useMemo } from "react";
import type { Side } from "../api";
import type { Rect } from "../geometry";
import { roundedPath, routeOrthogonal } from "../routing";

/** Rectangles of every block on the canvas (flow coordinates), for routing around them. */
export const ObstacleContext = createContext<Rect[]>([]);

/** Link drawn with orthogonal segments that never pass behind a block. */
function RoutedEdgeImpl(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, interactionWidth } = props;
  const obstacles = useContext(ObstacleContext);
  const path = useMemo(() => {
    const pts = routeOrthogonal(
      { x: sourceX, y: sourceY }, sourcePosition as Side,
      { x: targetX, y: targetY }, targetPosition as Side,
      obstacles,
    );
    if (pts) return roundedPath(pts);
    return getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 7 })[0];
  }, [sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, obstacles]);
  return <BaseEdge path={path} style={style} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 14} />;
}

export const RoutedEdge = memo(RoutedEdgeImpl);
