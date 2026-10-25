import { BaseEdge, getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { createContext, memo, useContext, useMemo } from "react";
import type { Side } from "../api";
import type { Point, Rect } from "../geometry";
import { roundedPath, routeOrthogonal } from "../routing";

/** Rectangles of every block on the canvas (flow coordinates), for routing around them. */
export const ObstacleContext = createContext<Rect[]>([]);

/** Paths of all the links, routed together (parallel links kept apart), by edge id. */
export const RouteContext = createContext<Map<string, Point[]>>(new Map());

/** Distance of a scope tap from its output port, along the port's straight stub. */
export const TAP_OFFSET = 12;
const STEP: Record<Side, [number, number]> = { left: [-1, 0], right: [1, 0], top: [0, -1], bottom: [0, 1] };

/**
 * Link drawn with orthogonal segments that never pass behind a block. A link feeding a
 * scope (``data.tap``) leaves from a solid junction dot on the output's own link, just
 * after the port: the scope taps the signal before it travels down the link.
 */
function RoutedEdgeImpl(props: EdgeProps) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, interactionWidth, data } = props;
  const obstacles = useContext(ObstacleContext);
  const shared = useContext(RouteContext).get(id);
  const tap = Boolean(data?.tap);
  const [dx, dy] = STEP[sourcePosition as Side] ?? [1, 0];
  const sx = tap ? sourceX + dx * TAP_OFFSET : sourceX;
  const sy = tap ? sourceY + dy * TAP_OFFSET : sourceY;
  const path = useMemo(() => {
    // The shared route, unless the handles moved since it was computed (mid-render of a drag).
    const near = (p: Point, x: number, y: number) => Math.abs(p.x - x) < 1.5 && Math.abs(p.y - y) < 1.5;
    if (shared && near(shared[0], sx, sy) && near(shared[shared.length - 1], targetX, targetY)) return roundedPath(shared);
    const pts = routeOrthogonal({ x: sx, y: sy }, sourcePosition as Side, { x: targetX, y: targetY }, targetPosition as Side, obstacles);
    if (pts) return roundedPath(pts);
    return getSmoothStepPath({ sourceX: sx, sourceY: sy, targetX, targetY, sourcePosition, targetPosition, borderRadius: 7 })[0];
  }, [shared, sx, sy, targetX, targetY, sourcePosition, targetPosition, obstacles]);
  return (
    <>
      <BaseEdge path={path} style={tap ? { ...style, strokeDasharray: "5 3" } : style} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 14} />
      {tap && (
        <>
          <path d={`M ${sourceX} ${sourceY} L ${sx} ${sy}`} style={{ stroke: style?.stroke, strokeWidth: 2 }} fill="none" />
          <circle className="tap-junction" cx={sx} cy={sy} r={4.5} style={{ fill: style?.stroke as string }} />
        </>
      )}
    </>
  );
}

export const RoutedEdge = memo(RoutedEdgeImpl);
