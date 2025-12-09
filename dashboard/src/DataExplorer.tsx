import { useState, useEffect } from 'react'
import { Search, Upload, Trash2, Save, X, FileText, Database, FolderOpen, AlertCircle, Shield, BarChart3 } from 'lucide-react'
import { CodeEditor } from './CodeEditor'

interface DataStatus {
  sourceFiles: {
    conversationsJson: boolean
    memoriesJson: boolean
  }
  generatedData: {
    conversations: boolean
    memory: boolean
    identityModel: boolean
  }
  counts: {
    conversationFiles: number
    memoryFiles: number
    files: number
  }
}

interface Conversation {
  id: string
  filename: string
  messageCount: number
  firstDate: string | null
  lastDate: string | null
  title: string
}

interface Memory {
  id: string
  type: string
  _file: string
  _preview: string
  [key: string]: any
}

type Tab = 'status' | 'conversations' | 'memories' | 'files' | 'identity'

interface IdentityModel {
  exists: boolean
  config?: any
  stylistic_profile?: any
  vocabulary_profile?: any
  identity_report?: string
}

interface FileItem {
  path: string
  name: string
  size: number
  modified: string
}

export function DataExplorer() {
  const [activeTab, setActiveTab] = useState<Tab>('status')
  const [status, setStatus] = useState<DataStatus | null>(null)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [memories, setMemories] = useState<Memory[]>([])
  const [files, setFiles] = useState<FileItem[]>([])
  const [identityModel, setIdentityModel] = useState<IdentityModel>({ exists: false })
  const [selectedItem, setSelectedItem] = useState<any>(null)
  const [editorContent, setEditorContent] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadingFiles, setUploadingFiles] = useState(false)

  useEffect(() => {
    loadStatus()
    loadConversations()
    loadMemories()
    loadFiles()
    loadIdentityModel()
    
    // Auto-refresh status and counts every 10 seconds
    const interval = setInterval(() => {
      loadStatus()
      // Also refresh counts for tabs
      loadConversations()
      loadMemories()
      loadFiles()
      loadIdentityModel()
    }, 10000)
    
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (activeTab === 'conversations') {
      loadConversations()
    } else if (activeTab === 'memories') {
      loadMemories()
    } else if (activeTab === 'files') {
      loadFiles()
    } else if (activeTab === 'identity') {
      loadIdentityModel()
    }
  }, [activeTab])

  const loadStatus = async () => {
    try {
      const res = await fetch('/api/mcp/data.status')
      const data = await res.json()
      setStatus(data)
    } catch (error) {
      console.error('Failed to load status:', error)
    }
  }

  const loadConversations = async () => {
    // Only show loading indicator if we're actively viewing the tab
    if (activeTab === 'conversations') {
      setLoading(true)
    }
    try {
      const res = await fetch('/api/mcp/data.conversations')
      const data = await res.json()
      setConversations(data.conversations || [])
    } catch (error) {
      console.error('Failed to load conversations:', error)
    } finally {
      if (activeTab === 'conversations') {
        setLoading(false)
      }
    }
  }

  const loadMemories = async () => {
    // Only show loading indicator if we're actively viewing the tab
    if (activeTab === 'memories') {
      setLoading(true)
    }
    try {
      const res = await fetch('/api/mcp/data.memories_list')
      const data = await res.json()
      setMemories(data.memories || [])
    } catch (error) {
      console.error('Failed to load memories:', error)
    } finally {
      if (activeTab === 'memories') {
        setLoading(false)
      }
    }
  }

  const loadFiles = async () => {
    if (activeTab === 'files') {
      setLoading(true)
    }
    try {
      const res = await fetch('/api/mcp/file.list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: 'files' })
      })
      const data = await res.json()
      
      // Transform the response into FileItem format
      const fileList: FileItem[] = (data.files || []).map((f: any) => ({
        path: f.path || f.filepath || f,
        name: (f.path || f.filepath || f).split('/').pop(),
        size: f.size || 0,
        modified: f.modified || new Date().toISOString()
      }))
      
      setFiles(fileList)
    } catch (error) {
      console.error('Failed to load files:', error)
      setFiles([])
    } finally {
      if (activeTab === 'files') {
        setLoading(false)
      }
    }
  }

  const loadIdentityModel = async () => {
    try {
      // Check if model exists
      const statusRes = await fetch('/api/mcp/identity_model.status')
      const statusData = await statusRes.json()
      
      if (!statusData.exists) {
        setIdentityModel({ exists: false })
        return
      }

      // Load model files
      const [configRes, stylisticRes, vocabRes, reportRes] = await Promise.all([
        fetch('/api/mcp/file.get', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filepath: 'models/identity/config.json' })
        }).catch(() => null),
        fetch('/api/mcp/file.get', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filepath: 'models/identity/stylistic_profile.json' })
        }).catch(() => null),
        fetch('/api/mcp/file.get', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filepath: 'models/identity/vocabulary_profile.json' })
        }).catch(() => null),
        fetch('/api/mcp/file.get', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filepath: 'memory/identity_report.md' })
        }).catch(() => null),
      ])

      const config = configRes ? await configRes.json().then(d => JSON.parse(d.content)) : null
      const stylistic = stylisticRes ? await stylisticRes.json().then(d => JSON.parse(d.content)) : null
      const vocab = vocabRes ? await vocabRes.json().then(d => JSON.parse(d.content)) : null
      const report = reportRes ? await reportRes.json().then(d => d.content) : null

      setIdentityModel({
        exists: true,
        config,
        stylistic_profile: stylistic,
        vocabulary_profile: vocab,
        identity_report: report
      })
    } catch (error) {
      console.error('Failed to load identity model:', error)
      setIdentityModel({ exists: false })
    }
  }

  const handleFilesUpload = async (fileList: FileList) => {
    setUploadingFiles(true)
    const successCount = { count: 0 }
    const failCount = { count: 0 }
    
    try {
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i]
        try {
          const text = await file.text()
          
          const res = await fetch('/api/mcp/file.upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              filename: file.name,
              content: text 
            })
          })
          
          if (res.ok) {
            successCount.count++
          } else {
            failCount.count++
          }
        } catch (error) {
          console.error(`Failed to upload ${file.name}:`, error)
          failCount.count++
        }
      }
      
      if (successCount.count > 0) {
        alert(`Successfully uploaded ${successCount.count} file(s)${failCount.count > 0 ? ` (${failCount.count} failed)` : ''}`)
        loadFiles()
        loadStatus()
      } else {
        alert('Failed to upload files')
      }
    } finally {
      setUploadingFiles(false)
    }
  }

  const handleFileDelete = async (filepath: string) => {
    if (!confirm(`Delete ${filepath}?`)) return
    
    try {
      const res = await fetch('/api/mcp/file.delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath })
      })
      
      if (res.ok) {
        alert('File deleted successfully')
        loadFiles()
        loadStatus()
      } else {
        alert('Failed to delete file')
      }
    } catch (error) {
      alert('Error deleting file')
    }
  }

  const openFile = async (filepath: string) => {
    try {
      const res = await fetch('/api/mcp/file.get', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath })
      })
      const data = await res.json()
      setSelectedItem({ type: 'file', filepath, title: filepath.split('/').pop() })
      setEditorContent(data.content || '')
    } catch (error) {
      alert('Failed to load file')
    }
  }

  const handleFileUpload = async (type: 'conversations' | 'memories', file: File) => {
    setLoading(true)
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const endpoint = type === 'conversations' ? '/api/mcp/data.upload_conversations' : '/api/mcp/data.upload_memories'
      
      // Show progress for large files
      const sizeInMB = file.size / (1024 * 1024)
      if (sizeInMB > 10) {
        console.log(`Uploading large file (${sizeInMB.toFixed(1)} MB), this may take a minute...`)
      }
      
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data })
      })
      
      if (res.ok) {
        alert(`${type}.json uploaded successfully!`)
        loadStatus()
      } else {
        const error = await res.text()
        alert(`Failed to upload ${type}.json: ${error}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert(`Error: ${error instanceof Error ? error.message : 'Invalid JSON file'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleClean = async (directory: string) => {
    if (!confirm(`Are you sure you want to clean ${directory}? This will remove all generated files.`)) {
      return
    }

    try {
      const res = await fetch('/api/mcp/data.clean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory })
      })
      
      const data = await res.json()
      alert(data.message)
      loadStatus()
      
      if (directory === 'conversations') {
        setConversations([])
      } else if (directory === 'memory') {
        setMemories([])
      }
    } catch (error) {
      alert('Failed to clean directory')
    }
  }

  const openConversation = async (conv: Conversation) => {
    try {
      const res = await fetch(`/api/mcp/data.conversation/${conv.id}`)
      const data = await res.json()
      setSelectedItem({ type: 'conversation', id: conv.id, title: conv.title })
      setEditorContent(data.content)
    } catch (error) {
      alert('Failed to load conversation')
    }
  }

  const openMemoryFile = async (filename: string) => {
    try {
      const res = await fetch(`/api/mcp/data.memory_file/${filename}`)
      const data = await res.json()
      setSelectedItem({ type: 'memory', filename, title: filename })
      setEditorContent(data.content)
    } catch (error) {
      alert('Failed to load memory file')
    }
  }

  const saveContent = async () => {
    if (!selectedItem) return

    try {
      let endpoint = ''
      let body = {}

      if (selectedItem.type === 'conversation') {
        endpoint = `/api/mcp/data.conversation/${selectedItem.id}`
        body = { content: editorContent }
      } else if (selectedItem.type === 'memory') {
        endpoint = `/api/mcp/data.memory_file/${selectedItem.filename}`
        body = { content: editorContent }
      } else if (selectedItem.type === 'file') {
        endpoint = `/api/mcp/file.upload`
        body = { filename: selectedItem.filepath.split('/').pop(), content: editorContent }
      } else if (selectedItem.type === 'report') {
        // Identity report is read-only
        alert('Identity report is read-only')
        return
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (res.ok) {
        alert('Saved successfully!')
      } else {
        alert('Failed to save')
      }
    } catch (error) {
      alert('Error saving content')
    }
  }

  const filteredConversations = conversations.filter(c =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.id.includes(searchQuery)
  )

  const filteredMemories = memories.filter(m =>
    m.type?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m._preview.toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Get unique memory files
  const memoryFiles = Array.from(new Set(memories.map(m => m._file)))

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-surface-200">
        <button
          onClick={() => setActiveTab('status')}
          className={`px-4 py-2 font-medium transition-colors border-b-2 ${
            activeTab === 'status'
              ? 'border-accent text-accent'
              : 'border-transparent text-text-secondary hover:text-text-primary'
          }`}
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Status & Upload
          </div>
        </button>
        <button
          onClick={() => setActiveTab('conversations')}
          className={`px-4 py-2 font-medium transition-colors border-b-2 ${
            activeTab === 'conversations'
              ? 'border-accent text-accent'
              : 'border-transparent text-text-secondary hover:text-text-primary'
          }`}
        >
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Conversations ({conversations.length})
          </div>
        </button>
        <button
          onClick={() => setActiveTab('memories')}
          className={`px-4 py-2 font-medium transition-colors border-b-2 ${
            activeTab === 'memories'
              ? 'border-accent text-accent'
              : 'border-transparent text-text-secondary hover:text-text-primary'
          }`}
        >
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Memories ({memories.length})
          </div>
        </button>
        <button
          onClick={() => setActiveTab('files')}
          className={`px-4 py-2 font-medium transition-colors border-b-2 ${
            activeTab === 'files'
              ? 'border-accent text-accent'
              : 'border-transparent text-text-secondary hover:text-text-primary'
          }`}
        >
          <div className="flex items-center gap-2">
            <FolderOpen className="w-4 h-4" />
            Files ({files.length})
          </div>
        </button>
        <button
          onClick={() => setActiveTab('identity')}
          className={`px-4 py-2 font-medium transition-colors border-b-2 ${
            activeTab === 'identity'
              ? 'border-accent text-accent'
              : 'border-transparent text-text-secondary hover:text-text-primary'
          }`}
        >
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4" />
            Identity Model
          </div>
        </button>
      </div>

      {/* Status Tab */}
      {activeTab === 'status' && status && (
        <div className="space-y-6">
          {/* Source Files */}
          <div className="card">
            <h3 className="font-display font-semibold text-text-primary mb-4">Source Files</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-surface-100">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">conversations.json</span>
                  <span className={`w-3 h-3 rounded-full ${status.sourceFiles.conversationsJson ? 'bg-success' : 'bg-danger'}`} />
                </div>
                <label className={`btn btn-secondary w-full ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
                  <Upload className="w-4 h-4" />
                  {loading ? 'Uploading...' : 'Upload'}
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    disabled={loading}
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) handleFileUpload('conversations', file)
                    }}
                  />
                </label>
                {loading && <div className="text-sm text-text-muted mt-2">Large files may take 1-2 minutes...</div>}
              </div>

              <div className="p-4 rounded-lg bg-surface-100">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">memories.json</span>
                  <span className={`w-3 h-3 rounded-full ${status.sourceFiles.memoriesJson ? 'bg-success' : 'bg-danger'}`} />
                </div>
                <label className={`btn btn-secondary w-full ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
                  <Upload className="w-4 h-4" />
                  {loading ? 'Uploading...' : 'Upload'}
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    disabled={loading}
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) handleFileUpload('memories', file)
                    }}
                  />
                </label>
                {loading && <div className="text-sm text-text-muted mt-2">Uploading...</div>}
              </div>
            </div>
          </div>

          {/* Generated Data */}
          <div className="card">
            <h3 className="font-display font-semibold text-text-primary mb-4">Generated Data</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-100 hover:bg-surface-200 transition-colors cursor-pointer group">
                <div onClick={() => setActiveTab('conversations')} className="flex-1">
                  <div className="font-medium group-hover:text-accent transition-colors">Conversations</div>
                  <div className="text-sm text-text-muted">{status.counts.conversationFiles} files</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${status.generatedData.conversations ? 'bg-success' : 'bg-surface-300'}`} />
                  <button onClick={(e) => { e.stopPropagation(); handleClean('conversations') }} className="btn btn-ghost text-danger">
                    <Trash2 className="w-4 h-4" />
                    Clean
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-100 hover:bg-surface-200 transition-colors cursor-pointer group">
                <div onClick={() => setActiveTab('memories')} className="flex-1">
                  <div className="font-medium group-hover:text-accent transition-colors">Memory Files</div>
                  <div className="text-sm text-text-muted">{status.counts.memoryFiles} files</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${status.generatedData.memory ? 'bg-success' : 'bg-surface-300'}`} />
                  <button onClick={(e) => { e.stopPropagation(); handleClean('memory') }} className="btn btn-ghost text-danger">
                    <Trash2 className="w-4 h-4" />
                    Clean
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-100 hover:bg-surface-200 transition-colors cursor-pointer group">
                <div onClick={() => setActiveTab('identity')} className="flex-1">
                  <div className="font-medium group-hover:text-accent transition-colors">Identity Model</div>
                  <div className="text-sm text-text-muted">Trained model</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`w-3 h-3 rounded-full ${status.generatedData.identityModel ? 'bg-success' : 'bg-surface-300'}`} />
                  <button onClick={(e) => { e.stopPropagation(); handleClean('models') }} className="btn btn-ghost text-danger">
                    <Trash2 className="w-4 h-4" />
                    Clean
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Conversations Tab */}
      {activeTab === 'conversations' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-muted" />
              <input
                type="text"
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 bg-surface-50 focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none"
              />
            </div>
          </div>

          {loading ? (
            <div className="text-center py-12 text-text-muted">Loading...</div>
          ) : (
            <div className="grid grid-cols-1 gap-3">
              {filteredConversations.map((conv) => (
                <div
                  key={conv.id}
                  onClick={() => openConversation(conv)}
                  className="card cursor-pointer hover:shadow-lg transition-shadow"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium text-text-primary mb-1">{conv.title}</h4>
                      <div className="flex items-center gap-4 text-sm text-text-muted">
                        <span>ID: {conv.id}</span>
                        <span>{conv.messageCount} messages</span>
                        {conv.firstDate && <span>{new Date(conv.firstDate).toLocaleDateString()}</span>}
                      </div>
                    </div>
                    <FileText className="w-5 h-5 text-accent" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Memories Tab */}
      {activeTab === 'memories' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-muted" />
              <input
                type="text"
                placeholder="Search memories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 bg-surface-50 focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {memoryFiles.map((filename) => (
              <div
                key={filename}
                onClick={() => openMemoryFile(filename)}
                className="card cursor-pointer hover:shadow-lg transition-shadow"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-text-primary mb-1">{filename}</h4>
                    <div className="text-sm text-text-muted">
                      {memories.filter(m => m._file === filename).length} records
                    </div>
                  </div>
                  <Database className="w-5 h-5 text-accent" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Identity Model Tab */}
      {activeTab === 'identity' && (
        <div className="space-y-6">
          {!identityModel.exists ? (
            <div className="card">
              <div className="text-center py-12 text-text-muted">
                <Shield className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="font-medium mb-2">No Identity Model Found</p>
                <p className="text-sm">Run "Train Identity Model" from the Pipeline to create your identity fingerprint</p>
              </div>
            </div>
          ) : (
            <>
              {/* Identity Report Summary */}
              {identityModel.identity_report && (
                <div className="card">
                  <h3 className="font-display font-semibold text-text-primary mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5 text-accent" />
                    Identity Analysis Report
                  </h3>
                  <div className="prose prose-sm max-w-none text-text-secondary bg-surface-100 p-4 rounded-lg overflow-auto max-h-96">
                    <pre className="whitespace-pre-wrap font-mono text-xs">{identityModel.identity_report}</pre>
                  </div>
                  <button 
                    onClick={() => {
                      setSelectedItem({ type: 'report', title: 'Identity Analysis Report' })
                      setEditorContent(identityModel.identity_report || '')
                    }}
                    className="btn btn-secondary mt-4"
                  >
                    <Eye className="w-4 h-4" />
                    View Full Report
                  </button>
                </div>
              )}

              {/* Model Configuration */}
              {identityModel.config && (
                <div className="card">
                  <h3 className="font-display font-semibold text-text-primary mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-accent" />
                    Model Configuration
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Base Model</div>
                      <div className="font-medium text-text-primary">{identityModel.config.base_model || 'N/A'}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Training Messages</div>
                      <div className="font-medium text-text-primary">{identityModel.config.training_messages?.toLocaleString() || 'N/A'}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Embedding Dimension</div>
                      <div className="font-medium text-text-primary">{identityModel.config.embedding_dim || 'N/A'}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Verification Threshold</div>
                      <div className="font-medium text-text-primary">{identityModel.config.verification_threshold || 'N/A'}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Created</div>
                      <div className="font-medium text-text-primary">
                        {identityModel.config.created_at ? new Date(identityModel.config.created_at).toLocaleDateString() : 'N/A'}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-100">
                      <div className="text-sm text-text-muted mb-1">Device</div>
                      <div className="font-medium text-text-primary">{identityModel.config.device || 'N/A'}</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Stylistic Profile */}
              {identityModel.stylistic_profile && (
                <div className="card">
                  <h3 className="font-display font-semibold text-text-primary mb-4">Stylistic Profile</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(identityModel.stylistic_profile).slice(0, 12).map(([key, value]: [string, any]) => (
                      <div key={key} className="p-3 rounded-lg bg-surface-100">
                        <div className="text-xs text-text-muted mb-1">{key.replace(/_/g, ' ')}</div>
                        <div className="font-medium text-text-primary text-sm">
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Top Vocabulary */}
              {identityModel.vocabulary_profile && (
                <div className="card">
                  <h3 className="font-display font-semibold text-text-primary mb-4">Top Vocabulary</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(identityModel.vocabulary_profile)
                      .sort(([, a]: [string, any], [, b]: [string, any]) => (b.identity_score || 0) - (a.identity_score || 0))
                      .slice(0, 50)
                      .map(([word, data]: [string, any]) => (
                        <div 
                          key={word} 
                          className="px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium"
                          title={`Score: ${data.identity_score?.toFixed(3) || 'N/A'}, Count: ${data.count || 0}`}
                        >
                          {word}
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Files Tab */}
      {activeTab === 'files' && (
        <div className="space-y-4">
          {/* Upload Section */}
          <div className="card">
            <h3 className="font-display font-semibold text-text-primary mb-4">Upload Files</h3>
            <label className={`btn btn-primary w-full md:w-auto ${uploadingFiles ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
              <Upload className="w-4 h-4" />
              {uploadingFiles ? 'Uploading...' : 'Choose Files to Upload'}
              <input
                type="file"
                multiple
                className="hidden"
                disabled={uploadingFiles}
                onChange={(e) => {
                  const files = e.target.files
                  if (files && files.length > 0) {
                    handleFilesUpload(files)
                  }
                }}
              />
            </label>
            <p className="text-sm text-text-muted mt-2">Upload text files (.txt, .md, .json, etc.) to the files directory for RAG access</p>
          </div>

          {/* Files List */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display font-semibold text-text-primary">Files ({files.length})</h3>
              <div className="relative flex-1 max-w-md ml-4">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-muted" />
                <input
                  type="text"
                  placeholder="Search files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 bg-surface-50 focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none"
                />
              </div>
            </div>

            {loading ? (
              <div className="text-center py-12 text-text-muted">Loading...</div>
            ) : files.length === 0 ? (
              <div className="text-center py-12 text-text-muted">
                <FolderOpen className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>No files uploaded yet</p>
                <p className="text-sm">Upload files to get started</p>
              </div>
            ) : (
              <div className="space-y-2">
                {files
                  .filter(f => f.name.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((file) => (
                    <div
                      key={file.path}
                      className="flex items-center justify-between p-3 rounded-lg bg-surface-100 hover:bg-surface-200 transition-colors group"
                    >
                      <div 
                        onClick={() => openFile(file.path)}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="font-medium text-text-primary group-hover:text-accent transition-colors">
                          {file.name}
                        </div>
                        <div className="text-sm text-text-muted">
                          {file.size > 0 ? `${(file.size / 1024).toFixed(1)} KB` : 'Unknown size'}
                          {file.modified && ` • ${new Date(file.modified).toLocaleDateString()}`}
                        </div>
                      </div>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation()
                          handleFileDelete(file.path)
                        }}
                        className="btn btn-ghost text-danger opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Editor Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-surface-50 rounded-xl border border-surface-200 w-full max-w-6xl max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-surface-200">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-accent" />
                <span className="font-medium text-text-primary">{selectedItem.title}</span>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={saveContent} className="btn btn-primary">
                  <Save className="w-4 h-4" />
                  Save
                </button>
                <button onClick={() => setSelectedItem(null)} className="btn btn-ghost">
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-hidden">
              <CodeEditor
                value={editorContent}
                onChange={(val) => setEditorContent(val || '')}
                language="json"
                height="calc(90vh - 80px)"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

