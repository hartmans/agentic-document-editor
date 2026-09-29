-- qwen_filter.lua

local function normalize_punctuation(s)
  if not s then return "" end
  s = s:gsub("‘", "'")
  s = s:gsub("’", "'")
  s = s:gsub("‚", "'")
  s = s:gsub("‛", "'")
  s = s:gsub("“", "\"")
  s = s:gsub("”", "\"")
  s = s:gsub("„", "\"")
  s = s:gsub("‟", "\"")
  s = s:gsub("—", "---")
  return s
end

-- Escape XML-sensitive characters in text nodes
local function escape_xml(s)
  if not s then return "" end
  s = normalize_punctuation(s)
  s = s:gsub("&", "&amp;")
  s = s:gsub("<", "«")
  s = s:gsub(">", "»")
  return s
end

-- Escape all ordinary text
function Str(el)
  return pandoc.Str(escape_xml(el.text))
end

-- Handle inline code elements
function Code(el)
  local role = nil
  if el.attr and el.attr.attributes then
    role = el.attr.attributes["role"]
  end

  local content = escape_xml(el.text)

  -- role = chat OR no role -> emit plain text without SSML
  if role == "chat" or role == nil then
    return pandoc.Str("`" .. content .. "`")
  end

  -- Any other role: ignore (emit plain escaped text)
  return pandoc.Str(content)
end

function Span(el)
  -- Check for class "title-ref"
  for _, class in ipairs(el.classes) do
    if class == "title-ref" then
      local text = pandoc.utils.stringify(el.content)
      return {
        pandoc.Str(escape_xml(text)),
        pandoc.RawInline("plain", "\n\n"),
      }
    end
  end

  -- Otherwise, leave unchanged
  return el
end

function Para(el)
  local text = normalize_punctuation(pandoc.utils.stringify(el.content))
  local _, double_quotes = text:gsub('"', '')
  if double_quotes % 2 == 1 then
    el.content:insert(pandoc.Str('"'))
  end
  return el
end
