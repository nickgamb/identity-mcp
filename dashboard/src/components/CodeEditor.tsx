import Editor from '@monaco-editor/react'

interface CodeEditorProps {
  value: string
  onChange: (value: string | undefined) => void
  language?: string
  readOnly?: boolean
  height?: string
}

export function CodeEditor({ value, onChange, language = 'json', readOnly = false, height = '600px' }: CodeEditorProps) {
  return (
    <Editor
      height={height}
      language={language}
      value={value}
      onChange={onChange}
      theme="vs-dark"
      options={{
        readOnly,
        minimap: { enabled: true },
        fontSize: 13,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        automaticLayout: true,
      }}
    />
  )
}

