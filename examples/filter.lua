-- ssml_roles.lua

-- Escape XML/SSML-sensitive characters in text nodes
local function escape_xml(s)
  if not s then return "" end
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

  -- role = skill → SSML emphasis
  if role == "skill" then
    return pandoc.RawInline(
      "plain",
      "<amazon:effect name=\"drc\">" .. content .. "</amazon:effect>")
  end

  -- role = chat OR no role → chat handling
  if role == "chat" or role == nil then
    return pandoc.RawInline(
      "plain",
      "<prosody rate=\"medium\">" .. content .. "</prosody>"
    )
  end

  -- Any other role: ignore (emit plain escaped text)
  return pandoc.Str(content)
end

function Span(el)
  -- Check for class "title-ref"
  for _, class in ipairs(el.classes) do
    if class == "title-ref" then
      -- Stringify span contents and XML-escape
      local text = pandoc.utils.stringify(el.content)
    return pandoc.RawInline(
      "plain",
      "<prosody rate=\"medium\">" .. escape_xml(text) .. "</prosody>"
    )
    end
  end

  -- Otherwise, leave unchanged
  return el
end
