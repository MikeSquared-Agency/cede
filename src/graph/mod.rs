use std::collections::HashMap;
use rusqlite::Connection;

use crate::error::Result;
use crate::types::*;
use crate::db::queries;

/// Breadth-first walk from a set of seed nodes, following edges in both
/// directions up to `depth` hops. Returns each discovered `NodeId` with
/// its minimum hop distance from any seed.
///
/// Designed to run inside a single `Db::call` closure (synchronous).
pub fn bfs_walk(
    conn: &Connection,
    seeds: &[NodeId],
    depth: usize,
) -> Result<HashMap<NodeId, usize>> {
    let mut visited: HashMap<NodeId, usize> = HashMap::new();
    let mut frontier: Vec<NodeId> = Vec::new();

    for id in seeds {
        visited.insert(id.clone(), 0);
        frontier.push(id.clone());
    }

    for hop in 1..=depth {
        let mut next_frontier = Vec::new();
        for node_id in &frontier {
            let neighbors = queries::get_neighbors(conn, node_id)?;
            for (neighbor_id, _edge_kind) in neighbors {
                if !visited.contains_key(&neighbor_id) {
                    visited.insert(neighbor_id.clone(), hop);
                    next_frontier.push(neighbor_id);
                }
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }

    Ok(visited)
}

/// Traverse from a single node — convenience wrapper.
pub fn traverse(
    conn: &Connection,
    start: &NodeId,
    depth: usize,
) -> Result<HashMap<NodeId, usize>> {
    bfs_walk(conn, &[start.clone()], depth)
}
