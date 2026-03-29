/// Keyboard and clipboard tools for theword.
///
/// These are registered into the `ToolRegistry` at startup and are available
/// to the dictation agent for typing text, sending shortcuts, and managing
/// the clipboard.
use std::sync::Arc;
use std::pin::Pin;
use std::future::Future;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use serde_json::json;

use crate::error::{CortexError, Result};
use crate::tools::{Tool, ToolRegistry};
use crate::types::ToolResult;

/// Register all keyboard/clipboard tools into the given registry.
pub fn register_keyboard_tools(registry: &mut ToolRegistry) {
    registry.register(make_type_text_tool());
    registry.register(make_press_shortcut_tool());
    registry.register(make_press_key_tool());
    registry.register(make_get_clipboard_tool());
    registry.register(make_set_clipboard_tool());
    registry.register(make_select_all_type_tool());
}

// ─── type_text ───────────────────────────────────────────

fn make_type_text_tool() -> Tool {
    Tool {
        name: "type_text".into(),
        description: "Type a string of text at the current cursor position. \
                      Use this to output cleaned dictation into whatever \
                      application is currently focused."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to type."
                }
            },
            "required": ["text"]
        }),
        trust: 0.9,
        handler: Arc::new(|input| {
            Box::pin(async move {
                let text = input["text"]
                    .as_str()
                    .ok_or_else(|| CortexError::Tool("type_text: missing 'text'".into()))?
                    .to_string();

                tokio::task::spawn_blocking(move || {
                    let mut enigo = Enigo::new(&Settings::default())
                        .map_err(|e| CortexError::Tool(format!("enigo init failed: {e}")))?;
                    enigo
                        .text(&text)
                        .map_err(|e| CortexError::Tool(format!("type_text failed: {e}")))?;
                    Ok::<_, CortexError>(())
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: "text typed".into(), success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── press_shortcut ──────────────────────────────────────

fn make_press_shortcut_tool() -> Tool {
    Tool {
        name: "press_shortcut".into(),
        description: "Send a keyboard shortcut. Specify as a '+'-separated list of \
                      modifiers and a key, e.g. 'ctrl+shift+t', 'cmd+c', 'alt+f4'. \
                      Supported modifiers: ctrl, shift, alt, cmd/meta/super."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "shortcut": {
                    "type": "string",
                    "description": "The shortcut string, e.g. 'ctrl+c', 'ctrl+shift+t'."
                }
            },
            "required": ["shortcut"]
        }),
        trust: 0.8,
        handler: Arc::new(|input| {
            Box::pin(async move {
                let shortcut = input["shortcut"]
                    .as_str()
                    .ok_or_else(|| CortexError::Tool("press_shortcut: missing 'shortcut'".into()))?
                    .to_lowercase();

                let parts: Vec<&str> = shortcut.split('+').collect();
                if parts.is_empty() {
                    return Err(CortexError::Tool("press_shortcut: empty shortcut".into()));
                }

                let (modifiers, key_str) = parts.split_at(parts.len() - 1);
                let key_str = key_str[0];

                let key = parse_key(key_str)?;
                let modifier_keys: Vec<Key> = modifiers
                    .iter()
                    .map(|m| parse_modifier(m))
                    .collect::<Result<Vec<_>>>()?;

                tokio::task::spawn_blocking(move || {
                    let mut enigo = Enigo::new(&Settings::default())
                        .map_err(|e| CortexError::Tool(format!("enigo init failed: {e}")))?;

                    for &m in &modifier_keys {
                        enigo.key(m, Direction::Press)
                            .map_err(|e| CortexError::Tool(format!("key press failed: {e}")))?;
                    }
                    enigo.key(key, Direction::Click)
                        .map_err(|e| CortexError::Tool(format!("key click failed: {e}")))?;
                    for &m in modifier_keys.iter().rev() {
                        enigo.key(m, Direction::Release)
                            .map_err(|e| CortexError::Tool(format!("key release failed: {e}")))?;
                    }
                    Ok::<_, CortexError>(())
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: format!("shortcut '{shortcut}' sent"), success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── press_key ───────────────────────────────────────────

fn make_press_key_tool() -> Tool {
    Tool {
        name: "press_key".into(),
        description: "Press and release a single key by name. \
                      Examples: 'Return', 'Tab', 'Escape', 'Up', 'Down', 'Left', 'Right', \
                      'Backspace', 'Delete', 'Home', 'End', 'PageUp', 'PageDown'."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key name to press."
                }
            },
            "required": ["key"]
        }),
        trust: 0.8,
        handler: Arc::new(|input| {
            Box::pin(async move {
                let key_str = input["key"]
                    .as_str()
                    .ok_or_else(|| CortexError::Tool("press_key: missing 'key'".into()))?
                    .to_string();

                let key = parse_key(&key_str.to_lowercase())?;

                tokio::task::spawn_blocking(move || {
                    let mut enigo = Enigo::new(&Settings::default())
                        .map_err(|e| CortexError::Tool(format!("enigo init failed: {e}")))?;
                    enigo
                        .key(key, Direction::Click)
                        .map_err(|e| CortexError::Tool(format!("press_key failed: {e}")))?;
                    Ok::<_, CortexError>(())
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: format!("key '{key_str}' pressed"), success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── get_clipboard ───────────────────────────────────────

fn make_get_clipboard_tool() -> Tool {
    Tool {
        name: "get_clipboard".into(),
        description: "Read the current contents of the system clipboard.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {}
        }),
        trust: 0.9,
        handler: Arc::new(|_input| {
            Box::pin(async move {
                let content = tokio::task::spawn_blocking(|| {
                    arboard::Clipboard::new()
                        .and_then(|mut cb| cb.get_text())
                        .map_err(|e| CortexError::Tool(format!("clipboard get failed: {e}")))
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: content, success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── set_clipboard ───────────────────────────────────────

fn make_set_clipboard_tool() -> Tool {
    Tool {
        name: "set_clipboard".into(),
        description: "Write text to the system clipboard.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to place on the clipboard."
                }
            },
            "required": ["text"]
        }),
        trust: 0.9,
        handler: Arc::new(|input| {
            Box::pin(async move {
                let text = input["text"]
                    .as_str()
                    .ok_or_else(|| CortexError::Tool("set_clipboard: missing 'text'".into()))?
                    .to_string();

                tokio::task::spawn_blocking(move || {
                    arboard::Clipboard::new()
                        .and_then(|mut cb| cb.set_text(text))
                        .map_err(|e| CortexError::Tool(format!("clipboard set failed: {e}")))
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: "clipboard set".into(), success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── select_all_type ─────────────────────────────────────

fn make_select_all_type_tool() -> Tool {
    Tool {
        name: "select_all_type".into(),
        description: "Select all text in the focused element (Ctrl+A / Cmd+A) \
                      then immediately type the given text, replacing the selection. \
                      Useful for rewriting the entire current text field."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The replacement text to type after selecting all."
                }
            },
            "required": ["text"]
        }),
        trust: 0.8,
        handler: Arc::new(|input| {
            Box::pin(async move {
                let text = input["text"]
                    .as_str()
                    .ok_or_else(|| CortexError::Tool("select_all_type: missing 'text'".into()))?
                    .to_string();

                tokio::task::spawn_blocking(move || {
                    let mut enigo = Enigo::new(&Settings::default())
                        .map_err(|e| CortexError::Tool(format!("enigo init failed: {e}")))?;

                    // Select all: Ctrl+A on Linux/Windows, Cmd+A on macOS
                    #[cfg(target_os = "macos")]
                    let select_mod = Key::Meta;
                    #[cfg(not(target_os = "macos"))]
                    let select_mod = Key::Control;

                    enigo.key(select_mod, Direction::Press)
                        .map_err(|e| CortexError::Tool(format!("modifier press failed: {e}")))?;
                    enigo.key(Key::Unicode('a'), Direction::Click)
                        .map_err(|e| CortexError::Tool(format!("select-all failed: {e}")))?;
                    enigo.key(select_mod, Direction::Release)
                        .map_err(|e| CortexError::Tool(format!("modifier release failed: {e}")))?;

                    // Small delay to ensure selection registers
                    std::thread::sleep(std::time::Duration::from_millis(50));

                    enigo.text(&text)
                        .map_err(|e| CortexError::Tool(format!("type failed: {e}")))?;

                    Ok::<_, CortexError>(())
                })
                .await
                .map_err(|e| CortexError::Tool(format!("spawn_blocking failed: {e}")))??;

                Ok(ToolResult { output: "selected all and typed".into(), success: true })
            }) as Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
        }),
    }
}

// ─── Key parsing helpers ─────────────────────────────────

fn parse_key(s: &str) -> Result<Key> {
    let k = match s {
        "return" | "enter"  => Key::Return,
        "tab"               => Key::Tab,
        "escape" | "esc"    => Key::Escape,
        "backspace"         => Key::Backspace,
        "delete" | "del"    => Key::Delete,
        "home"              => Key::Home,
        "end"               => Key::End,
        "pageup"            => Key::PageUp,
        "pagedown"          => Key::PageDown,
        "up"                => Key::UpArrow,
        "down"              => Key::DownArrow,
        "left"              => Key::LeftArrow,
        "right"             => Key::RightArrow,
        "space"             => Key::Space,
        "f1"  => Key::F1,  "f2"  => Key::F2,  "f3"  => Key::F3,  "f4"  => Key::F4,
        "f5"  => Key::F5,  "f6"  => Key::F6,  "f7"  => Key::F7,  "f8"  => Key::F8,
        "f9"  => Key::F9,  "f10" => Key::F10, "f11" => Key::F11, "f12" => Key::F12,
        other => {
            let mut chars = other.chars();
            if let Some(c) = chars.next() {
                if chars.next().is_none() {
                    Key::Unicode(c)
                } else {
                    return Err(CortexError::Tool(format!("unknown key: '{other}'")));
                }
            } else {
                return Err(CortexError::Tool("empty key string".into()));
            }
        }
    };
    Ok(k)
}

fn parse_modifier(s: &str) -> Result<Key> {
    match s {
        "ctrl" | "control"           => Ok(Key::Control),
        "shift"                      => Ok(Key::Shift),
        "alt"                        => Ok(Key::Alt),
        "cmd" | "meta" | "super"     => Ok(Key::Meta),
        other => Err(CortexError::Tool(format!("unknown modifier: '{other}'"))),
    }
}
