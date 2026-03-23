use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::types::*;

// ─── Node CRUD ──────────────────────────────────────────

pub fn insert_node(conn: &Connection, node: &Node) -> Result<()> {
    let embedding_blob: Option<Vec<u8>> = node.embedding.as_ref().map(|v| {
        bytemuck::cast_slice::<f32, u8>(v).to_vec()
    });
    conn.execute(
        "INSERT INTO nodes (id, kind, title, body, importance, trust_score,
                            access_count, created_at, last_access, decay_rate, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            node.id,
            node.kind.as_str(),
            node.title,
            node.body,
            node.importance,
            node.trust_score,
            node.access_count,
            node.created_at,
            node.last_access,
            node.decay_rate,
            embedding_blob,
        ],
    )?;
    Ok(())
}

pub fn get_node(conn: &Connection, id: &str) -> Result<Option<Node>> {
    conn.query_row(
        "SELECT id, kind, title, body, importance, trust_score,
                access_count, created_at, last_access, decay_rate, embedding
         FROM nodes WHERE id = ?1",
        params![id],
        |row| {
            let embedding_blob: Option<Vec<u8>> = row.get(10)?;
            let embedding = embedding_blob.map(|b| blob_to_embedding(&b));
            let kind_str: String = row.get(1)?;
            Ok(Node {
                id: row.get(0)?,
                kind: NodeKind::from_str_opt(&kind_str).unwrap_or(NodeKind::Fact),
                title: row.get(2)?,
                body: row.get(3)?,
                importance: row.get(4)?,
                trust_score: row.get(5)?,
                access_count: row.get(6)?,
                created_at: row.get(7)?,
                last_access: row.get(8)?,
                decay_rate: row.get(9)?,
                embedding,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn get_nodes_by_ids(conn: &Connection, ids: &HashSet<NodeId>) -> Result<Vec<Node>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: Vec<String> = ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 1)).collect();
    let sql = format!(
        "SELECT id, kind, title, body, importance, trust_score,
                access_count, created_at, last_access, decay_rate, embedding
         FROM nodes WHERE id IN ({})",
        placeholders.join(", ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let id_vec: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    let params: Vec<&dyn rusqlite::types::ToSql> = id_vec
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(&*params, |row| {
        let embedding_blob: Option<Vec<u8>> = row.get(10)?;
        let embedding = embedding_blob.map(|b| blob_to_embedding(&b));
        let kind_str: String = row.get(1)?;
        Ok(Node {
            id: row.get(0)?,
            kind: NodeKind::from_str_opt(&kind_str).unwrap_or(NodeKind::Fact),
            title: row.get(2)?,
            body: row.get(3)?,
            importance: row.get(4)?,
            trust_score: row.get(5)?,
            access_count: row.get(6)?,
            created_at: row.get(7)?,
            last_access: row.get(8)?,
            decay_rate: row.get(9)?,
            embedding,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

pub fn get_nodes_by_kind(conn: &Connection, kind: NodeKind) -> Result<Vec<Node>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, title, body, importance, trust_score,
                access_count, created_at, last_access, decay_rate, embedding
         FROM nodes WHERE kind = ?1",
    )?;
    let rows = stmt.query_map(params![kind.as_str()], |row| {
        let embedding_blob: Option<Vec<u8>> = row.get(10)?;
        let embedding = embedding_blob.map(|b| blob_to_embedding(&b));
        let kind_str: String = row.get(1)?;
        Ok(Node {
            id: row.get(0)?,
            kind: NodeKind::from_str_opt(&kind_str).unwrap_or(NodeKind::Fact),
            title: row.get(2)?,
            body: row.get(3)?,
            importance: row.get(4)?,
            trust_score: row.get(5)?,
            access_count: row.get(6)?,
            created_at: row.get(7)?,
            last_access: row.get(8)?,
            decay_rate: row.get(9)?,
            embedding,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

pub fn update_node_importance(conn: &Connection, id: &str, importance: f64) -> Result<()> {
    conn.execute(
        "UPDATE nodes SET importance = ?1 WHERE id = ?2",
        params![importance, id],
    )?;
    Ok(())
}

pub fn update_node_trust(conn: &Connection, id: &str, trust: f64) -> Result<()> {
    conn.execute(
        "UPDATE nodes SET trust_score = ?1 WHERE id = ?2",
        params![trust, id],
    )?;
    Ok(())
}

pub fn touch_nodes(conn: &Connection, ids: &[NodeId]) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let mut stmt = conn.prepare(
        "UPDATE nodes SET last_access = ?1, access_count = access_count + 1 WHERE id = ?2",
    )?;
    for id in ids {
        stmt.execute(params![now, id])?;
    }
    Ok(())
}

/// Load every (NodeId, embedding) pair for HNSW rebuild.
pub fn get_all_embeddings(conn: &Connection) -> Result<Vec<(NodeId, Vec<f32>)>> {
    let mut stmt = conn.prepare(
        "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL",
    )?;
    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        Ok((id, blob_to_embedding(&blob)))
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

/// Get stale nodes eligible for decay (not accessed in >24h, decay_rate > 0).
pub fn get_decayable_nodes(conn: &Connection) -> Result<Vec<(NodeId, f64, f64)>> {
    let cutoff = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        - 86400; // 24 hours
    let mut stmt = conn.prepare(
        "SELECT id, importance, decay_rate FROM nodes
         WHERE decay_rate > 0.0
           AND (last_access IS NULL OR last_access < ?1)
           AND importance > 0.01",
    )?;
    let rows = stmt.query_map(params![cutoff], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?))
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

// ─── Edge CRUD ──────────────────────────────────────────

pub fn insert_edge(conn: &Connection, edge: &Edge) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO edges (id, src, dst, kind, weight, created_at, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            edge.id,
            edge.src,
            edge.dst,
            edge.kind.as_str(),
            edge.weight,
            edge.created_at,
            edge.metadata,
        ],
    )?;
    Ok(())
}

pub fn get_edges_from(conn: &Connection, src: &str) -> Result<Vec<Edge>> {
    let mut stmt = conn.prepare(
        "SELECT id, src, dst, kind, weight, created_at, metadata
         FROM edges WHERE src = ?1",
    )?;
    read_edges(&mut stmt, params![src])
}

pub fn get_edges_to(conn: &Connection, dst: &str) -> Result<Vec<Edge>> {
    let mut stmt = conn.prepare(
        "SELECT id, src, dst, kind, weight, created_at, metadata
         FROM edges WHERE dst = ?1",
    )?;
    read_edges(&mut stmt, params![dst])
}

pub fn edge_exists(conn: &Connection, src: &str, dst: &str, kind: EdgeKind) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND kind = ?3",
        params![src, dst, kind.as_str()],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Get all neighbors of a node (both directions).
pub fn get_neighbors(conn: &Connection, node_id: &str) -> Result<Vec<(NodeId, EdgeKind)>> {
    let mut stmt = conn.prepare(
        "SELECT dst, kind FROM edges WHERE src = ?1
         UNION
         SELECT src, kind FROM edges WHERE dst = ?1",
    )?;
    let rows = stmt.query_map(params![node_id], |row| {
        let id: String = row.get(0)?;
        let kind_str: String = row.get(1)?;
        Ok((id, EdgeKind::from_str_opt(&kind_str).unwrap_or(EdgeKind::RelatesTo)))
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

// ─── Contradictions ─────────────────────────────────────

pub fn insert_contradiction(conn: &Connection, node_a: &str, node_b: &str) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    conn.execute(
        "INSERT OR IGNORE INTO contradictions (node_a, node_b, detected_at, resolved)
         VALUES (?1, ?2, ?3, 0)",
        params![node_a, node_b, now],
    )?;
    Ok(())
}

pub fn get_unresolved_contradictions(
    conn: &Connection,
    node_ids: &[NodeId],
) -> Result<Vec<ContradictionPair>> {
    if node_ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: Vec<String> = node_ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect();
    let set = placeholders.join(", ");
    let sql = format!(
        "SELECT node_a, node_b, detected_at, resolved FROM contradictions
         WHERE resolved = 0 AND (node_a IN ({set}) OR node_b IN ({set}))"
    );
    let mut stmt = conn.prepare(&sql)?;
    let id_vec: Vec<&str> = node_ids.iter().map(|s| s.as_str()).collect();
    // SQLite reuses ?1, ?2, etc. across both IN clauses, so pass params once
    let params: Vec<&dyn rusqlite::types::ToSql> = id_vec
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(&*params, |row| {
        Ok(ContradictionPair {
            node_a: row.get(0)?,
            node_b: row.get(1)?,
            detected_at: row.get(2)?,
            resolved: row.get::<_, i64>(3)? != 0,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

pub fn get_all_unresolved_contradictions(conn: &Connection) -> Result<Vec<ContradictionPair>> {
    let mut stmt = conn.prepare(
        "SELECT node_a, node_b, detected_at, resolved FROM contradictions WHERE resolved = 0",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ContradictionPair {
            node_a: row.get(0)?,
            node_b: row.get(1)?,
            detected_at: row.get(2)?,
            resolved: false,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

pub fn resolve_contradiction(conn: &Connection, node_a: &str, node_b: &str) -> Result<()> {
    conn.execute(
        "UPDATE contradictions SET resolved = 1 WHERE node_a = ?1 AND node_b = ?2",
        params![node_a, node_b],
    )?;
    Ok(())
}

// ─── Meta ───────────────────────────────────────────────

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    conn.query_row(
        "SELECT value FROM meta WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

// ─── Stats ──────────────────────────────────────────────

pub fn node_count(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))?)
}

pub fn edge_count(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?)
}

pub fn node_count_by_kind(conn: &Connection) -> Result<HashMap<String, i64>> {
    let mut stmt = conn.prepare("SELECT kind, COUNT(*) FROM nodes GROUP BY kind")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    let mut map = HashMap::new();
    for r in rows {
        let (k, v) = r?;
        map.insert(k, v);
    }
    Ok(map)
}

/// Get edges with incoming Supports to a node (from trusted sources).
pub fn count_supporting_edges(conn: &Connection, node_id: &str, min_trust: f64) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM edges e
         JOIN nodes n ON e.src = n.id
         WHERE e.dst = ?1 AND e.kind = 'supports' AND n.trust_score > ?2",
        params![node_id, min_trust],
        |row| row.get(0),
    )?)
}

/// Count unresolved contradictions involving a node.
pub fn count_contradictions(conn: &Connection, node_id: &str) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM contradictions
         WHERE resolved = 0 AND (node_a = ?1 OR node_b = ?1)",
        params![node_id],
        |row| row.get(0),
    )?)
}

// ─── Helpers ────────────────────────────────────────────

fn read_edges(
    stmt: &mut rusqlite::Statement,
    params: impl rusqlite::Params,
) -> Result<Vec<Edge>> {
    let rows = stmt.query_map(params, |row| {
        let kind_str: String = row.get(3)?;
        Ok(Edge {
            id: row.get(0)?,
            src: row.get(1)?,
            dst: row.get(2)?,
            kind: EdgeKind::from_str_opt(&kind_str).unwrap_or(EdgeKind::RelatesTo),
            weight: row.get(4)?,
            created_at: row.get(5)?,
            metadata: row.get(6)?,
        })
    })?;
    let mut result = Vec::new();
    for r in rows {
        result.push(r?);
    }
    Ok(result)
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 {
        return vec![];
    }
    bytemuck::cast_slice::<u8, f32>(blob).to_vec()
}
