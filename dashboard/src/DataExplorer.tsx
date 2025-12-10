import { useState, useEffect } from 'react'
import { Upload, Trash2, Save, X, FileText, Database, FolderOpen, AlertCircle, Shield, BarChart3, Eye, TrendingUp, TrendingDown, Activity } from 'lucide-react'
import { CodeEditor } from './components/CodeEditor'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, LineChart, Area, AreaChart, ReferenceLine } from 'recharts'
import { SearchInput } from './components/SearchInput'
import { EmptyState } from './components/EmptyState'
import { CollapsibleSection } from './components/CollapsibleSection'
import { WordCloud } from './components/WordCloud'

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
  temporal_analysis?: any
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
  const [vocabTab, setVocabTab] = useState<'frequencies' | 'distinctive' | 'bigrams' | 'boosted'>('frequencies')

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
      // Get model status and data (includes config, profiles, and report)
      const statusRes = await fetch('/api/mcp/identity.model_status')
      if (!statusRes.ok) {
        throw new Error(`HTTP ${statusRes.status}: ${statusRes.statusText}`)
      }
      const statusData = await statusRes.json()
      
      if (!statusData.exists) {
        setIdentityModel({ exists: false })
        return
      }

      setIdentityModel({
        exists: true,
        config: statusData.config || null,
        stylistic_profile: statusData.stylistic_profile || null,
        vocabulary_profile: statusData.vocabulary_profile || null,
        temporal_analysis: statusData.temporal_analysis || null,
        identity_report: statusData.identity_report || null
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
        setIdentityModel({ exists: false })
      } else if (directory === 'models') {
        setIdentityModel({ exists: false })
      }
      
      // Notify parent to refresh pipeline status
      window.dispatchEvent(new CustomEvent('data-cleaned', { detail: { directory } }))
    } catch (error) {
      alert('Failed to clean directory')
    }
  }

  const handleDeleteSourceFile = async (type: 'conversations' | 'memories') => {
    const filename = type === 'conversations' ? 'conversations.json' : 'memories.json'
    
    if (!confirm(`Delete ${filename}? This will remove the source file.`)) {
      return
    }

    try {
      const res = await fetch('/api/mcp/data.delete_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type })
      })
      
      const data = await res.json()
      
      if (data.success) {
        alert(`${filename} deleted successfully`)
        loadStatus()
        window.dispatchEvent(new CustomEvent('data-cleaned', { detail: { directory: type } }))
      } else {
        alert(`Failed to delete ${filename}: ${data.message || 'Unknown error'}`)
      }
    } catch (error) {
      alert('Error deleting file')
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
                <div className="flex gap-2">
                  <label className={`btn btn-secondary flex-1 ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
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
                  {status.sourceFiles.conversationsJson && (
                    <button 
                      onClick={() => handleDeleteSourceFile('conversations')}
                      className="btn btn-ghost text-danger"
                      disabled={loading}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
                {loading && <div className="text-sm text-text-muted mt-2">Large files may take 1-2 minutes...</div>}
              </div>

              <div className="p-4 rounded-lg bg-surface-100">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">memories.json</span>
                  <span className={`w-3 h-3 rounded-full ${status.sourceFiles.memoriesJson ? 'bg-success' : 'bg-danger'}`} />
                </div>
                <div className="flex gap-2">
                  <label className={`btn btn-secondary flex-1 ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
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
                  {status.sourceFiles.memoriesJson && (
                    <button 
                      onClick={() => handleDeleteSourceFile('memories')}
                      className="btn btn-ghost text-danger"
                      disabled={loading}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
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
            <SearchInput
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {loading ? (
            <div className="text-center py-12 text-text-muted">Loading...</div>
          ) : filteredConversations.length === 0 ? (
            <EmptyState
              icon={FileText}
              title="No conversations found"
              message={searchQuery ? "Try a different search term" : "No conversations available"}
            />
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
            <SearchInput
              placeholder="Search memories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
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
              <EmptyState
                icon={Shield}
                title="No Identity Model Found"
                message='Run "Train Identity Model" from the Pipeline to create your identity fingerprint'
              />
            </div>
          ) : (
            <>
              {/* Delete Button */}
              <div className="card">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-display font-semibold text-text-primary mb-1">Identity Model</h3>
                    <p className="text-sm text-text-muted">Trained identity fingerprint for verification</p>
                  </div>
                  <button 
                    onClick={() => {
                      if (confirm('Are you sure you want to delete the identity model? This will remove all model files from models/identity/')) {
                        handleClean('models')
                      }
                    }}
                    className="btn btn-ghost text-danger"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Model
                  </button>
                </div>
              </div>

              {/* Model Quality Insights */}
              {identityModel.config && (
                <div className="card">
                  <h3 className="font-display font-semibold text-text-primary mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-accent" />
                    Model Quality & Insights
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Training Data Quality</div>
                      <div className="text-2xl font-bold text-text-primary mb-1">
                        {identityModel.config.num_messages >= 1000 ? 'Excellent' : 
                         identityModel.config.num_messages >= 500 ? 'Good' : 
                         identityModel.config.num_messages >= 100 ? 'Fair' : 'Limited'}
                      </div>
                      <div className="text-sm text-text-muted">
                        {identityModel.config.num_messages?.toLocaleString() || 0} messages, {identityModel.config.num_conversations || 0} conversations
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Model Stability</div>
                      <div className="text-2xl font-bold text-text-primary mb-1">
                        {identityModel.config.statistics?.std_similarity < 0.1 ? 'High' :
                         identityModel.config.statistics?.std_similarity < 0.15 ? 'Medium' : 'Variable'}
                      </div>
                      <div className="text-sm text-text-muted">
                        Std: {identityModel.config.statistics?.std_similarity?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Verification Confidence</div>
                      <div className="text-2xl font-bold text-text-primary mb-1">
                        {identityModel.config.statistics?.mean_similarity > 0.85 ? 'High' :
                         identityModel.config.statistics?.mean_similarity > 0.75 ? 'Medium' : 'Low'}
                      </div>
                      <div className="text-sm text-text-muted">
                        Mean: {identityModel.config.statistics?.mean_similarity?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                  </div>
                  {identityModel.temporal_analysis && identityModel.temporal_analysis.evolution && (
                    <div className="mt-4 p-3 rounded-lg bg-surface-100">
                      <div className="flex items-center gap-2 text-sm">
                        {identityModel.temporal_analysis.evolution.similarity_trend > 0 ? (
                          <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                        <span className="text-text-muted">Identity Evolution:</span>
                        <span className="font-medium text-text-primary">
                          {identityModel.temporal_analysis.evolution.similarity_trend > 0 ? 'Strengthening' : 'Stable'} over time
                          {identityModel.temporal_analysis.time_span_days && (
                            <span className="text-text-muted ml-2">
                              ({identityModel.temporal_analysis.time_span_days.toFixed(1)} days)
                            </span>
                          )}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Temporal Evolution Chart */}
              {identityModel.temporal_analysis && identityModel.temporal_analysis.windows && identityModel.temporal_analysis.windows.length > 0 && (
                <CollapsibleSection 
                  title="Identity Evolution Over Time" 
                  icon={Activity}
                  defaultExpanded={true}
                >
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={identityModel.temporal_analysis.windows.map((w: any, i: number) => ({
                          period: `Period ${i + 1}`,
                          start: new Date(w.start_time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                          mean_similarity: w.mean_similarity,
                          std_similarity: w.std_similarity,
                          message_count: w.message_count
                        }))}
                        margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis 
                          dataKey="start" 
                          stroke="rgba(255,255,255,0.5)"
                          angle={-45}
                          textAnchor="end"
                          height={80}
                        />
                        <YAxis 
                          stroke="rgba(255,255,255,0.5)"
                          label={{ value: 'Similarity', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                          labelStyle={{ color: '#fff' }}
                          formatter={(value: any, name: string, props: any) => {
                            if (name === 'mean_similarity') {
                              const std = props.payload?.std_similarity || 0
                              const count = props.payload?.message_count || 0
                              return [
                                `${value.toFixed(4)} ± ${std.toFixed(4)} (${count} msgs)`,
                                'Mean Similarity'
                              ]
                            }
                            if (name === 'std_similarity') {
                              return [`±${value.toFixed(4)}`, 'Std Dev']
                            }
                            return [value, name]
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="mean_similarity" 
                          stroke="#8b5cf6" 
                          strokeWidth={2}
                          dot={{ fill: '#8b5cf6', r: 4 }}
                          name="Mean Similarity"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="std_similarity" 
                          stroke="#a78bfa" 
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={{ fill: '#a78bfa', r: 2 }}
                          name="Std Dev"
                        />
                        {identityModel.config?.statistics?.similarity_threshold_1std && (
                          <ReferenceLine 
                            y={identityModel.config.statistics.similarity_threshold_1std} 
                            stroke="#f59e0b" 
                            strokeDasharray="5 5"
                            label={{ value: '1σ Threshold', position: 'right', fill: '#f59e0b' }}
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-sm text-text-muted space-y-1">
                    <div>Time span: {identityModel.temporal_analysis.time_span_days?.toFixed(1)} days</div>
                    <div>First message: {new Date(identityModel.temporal_analysis.first_message).toLocaleDateString()}</div>
                    <div>Last message: {new Date(identityModel.temporal_analysis.last_message).toLocaleDateString()}</div>
                    {identityModel.temporal_analysis.evolution && (
                      <div className="flex gap-4 mt-2">
                        <span>Early period: {identityModel.temporal_analysis.evolution.early_mean_similarity?.toFixed(4)}</span>
                        <span>Late period: {identityModel.temporal_analysis.evolution.late_mean_similarity?.toFixed(4)}</span>
                        <span className={identityModel.temporal_analysis.evolution.similarity_trend > 0 ? 'text-green-500' : 'text-red-500'}>
                          Trend: {identityModel.temporal_analysis.evolution.similarity_trend > 0 ? '+' : ''}{identityModel.temporal_analysis.evolution.similarity_trend?.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                </CollapsibleSection>
              )}

              {/* Similarity Distribution Chart */}
              {identityModel.config?.statistics && (
                <CollapsibleSection 
                  title="Similarity Distribution" 
                  icon={BarChart3}
                  defaultExpanded={true}
                  detailsContent={
                    <div className="space-y-2 text-sm">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-text-muted mb-1">Percentiles</div>
                          <div className="space-y-1">
                            <div>P25: {identityModel.config.statistics.percentiles?.p25?.toFixed(4) || 'N/A'}</div>
                            <div>P50: {identityModel.config.statistics.percentiles?.p50?.toFixed(4) || 'N/A'}</div>
                            <div>P75: {identityModel.config.statistics.percentiles?.p75?.toFixed(4) || 'N/A'}</div>
                            <div>P90: {identityModel.config.statistics.percentiles?.p90?.toFixed(4) || 'N/A'}</div>
                            <div>P95: {identityModel.config.statistics.percentiles?.p95?.toFixed(4) || 'N/A'}</div>
                          </div>
                        </div>
                        <div>
                          <div className="text-text-muted mb-1">Distance Metrics</div>
                          <div className="space-y-1">
                            <div>Mean: {identityModel.config.statistics.mean_distance?.toFixed(4) || 'N/A'}</div>
                            <div>Std: {identityModel.config.statistics.std_distance?.toFixed(4) || 'N/A'}</div>
                            <div>Min: {identityModel.config.statistics.min_distance?.toFixed(4) || 'N/A'}</div>
                            <div>Max: {identityModel.config.statistics.max_distance?.toFixed(4) || 'N/A'}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  }
                >
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={(() => {
                          // Generate distribution data from mean and std
                          const mean = identityModel.config.statistics.mean_similarity || 0;
                          const std = identityModel.config.statistics.std_similarity || 0.1;
                          
                          // Generate points for normal distribution curve
                          const points = [];
                          const min = Math.max(0, mean - 3 * std);
                          const max = Math.min(1, mean + 3 * std);
                          for (let i = 0; i <= 50; i++) {
                            const x = min + (max - min) * (i / 50);
                            // Normal distribution PDF approximation
                            const y = Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
                            points.push({ similarity: x.toFixed(3), density: y });
                          }
                          return points;
                        })()}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis 
                          dataKey="similarity" 
                          stroke="rgba(255,255,255,0.5)"
                          label={{ value: 'Similarity Score', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.7)' }}
                        />
                        <YAxis 
                          stroke="rgba(255,255,255,0.5)"
                          label={{ value: 'Density', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                          labelStyle={{ color: '#fff' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="density" 
                          stroke="#8b5cf6" 
                          fill="#8b5cf6" 
                          fillOpacity={0.3}
                        />
                        {identityModel.config.statistics.similarity_threshold_1std && (
                          <ReferenceLine 
                            x={identityModel.config.statistics.similarity_threshold_1std} 
                            stroke="#f59e0b" 
                            strokeDasharray="5 5"
                            label={{ value: '1σ', position: 'top', fill: '#f59e0b' }}
                          />
                        )}
                        {identityModel.config.statistics.similarity_threshold_2std && (
                          <ReferenceLine 
                            x={identityModel.config.statistics.similarity_threshold_2std} 
                            stroke="#ef4444" 
                            strokeDasharray="5 5"
                            label={{ value: '2σ', position: 'top', fill: '#ef4444' }}
                          />
                        )}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 text-sm text-text-muted space-y-1">
                    <div>Mean: {identityModel.config.statistics.mean_similarity?.toFixed(4)} ± {identityModel.config.statistics.std_similarity?.toFixed(4)}</div>
                    <div className="flex gap-4">
                      <span>1σ Threshold: <span className="text-amber-500">{identityModel.config.statistics.similarity_threshold_1std?.toFixed(4)}</span></span>
                      <span>2σ Threshold: <span className="text-red-500">{identityModel.config.statistics.similarity_threshold_2std?.toFixed(4)}</span></span>
                    </div>
                  </div>
                </CollapsibleSection>
              )}

              {/* Model Configuration */}
              {identityModel.config && (
                <CollapsibleSection 
                  title="Model Configuration" 
                  icon={BarChart3}
                  detailsContent={
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Model Name</div>
                        <div className="font-medium text-text-primary">{identityModel.config.model_name || identityModel.config.base_model || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Training Messages</div>
                        <div className="font-medium text-text-primary">{identityModel.config.num_messages?.toLocaleString() || identityModel.config.training_messages?.toLocaleString() || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Conversations</div>
                        <div className="font-medium text-text-primary">{identityModel.config.num_conversations?.toLocaleString() || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Embedding Dimension</div>
                        <div className="font-medium text-text-primary">{identityModel.config.statistics?.embedding_dim || identityModel.config.embedding_dim || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Verification Threshold (1σ)</div>
                        <div className="font-medium text-text-primary">{identityModel.config.statistics?.similarity_threshold_1std?.toFixed(4) || identityModel.config.verification_threshold || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Verification Threshold (2σ)</div>
                        <div className="font-medium text-text-primary">{identityModel.config.statistics?.similarity_threshold_2std?.toFixed(4) || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Mean Similarity</div>
                        <div className="font-medium text-text-primary">{identityModel.config.statistics?.mean_similarity?.toFixed(4) || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Std Similarity</div>
                        <div className="font-medium text-text-primary">{identityModel.config.statistics?.std_similarity?.toFixed(4) || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Created</div>
                        <div className="font-medium text-text-primary">
                          {identityModel.config.created_at ? new Date(identityModel.config.created_at).toLocaleDateString() : 'N/A'}
                        </div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Model Size</div>
                        <div className="font-medium text-text-primary">{identityModel.config.model_size || 'N/A'}</div>
                      </div>
                      <div className="p-3 rounded-lg bg-surface-100">
                        <div className="text-sm text-text-muted mb-1">Device Trained On</div>
                        <div className="font-medium text-text-primary">{identityModel.config.device_trained_on || identityModel.config.device || 'N/A'}</div>
                      </div>
                      {identityModel.config.statistics?.num_samples && (
                        <div className="p-3 rounded-lg bg-surface-100">
                          <div className="text-sm text-text-muted mb-1">Samples</div>
                          <div className="font-medium text-text-primary">{identityModel.config.statistics.num_samples.toLocaleString()}</div>
                        </div>
                      )}
                    </div>
                  }
                >
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Base Model</div>
                      <div className="font-semibold text-text-primary">{identityModel.config.model_name || identityModel.config.base_model || 'N/A'}</div>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Training Data</div>
                      <div className="font-semibold text-text-primary">
                        {identityModel.config.num_messages?.toLocaleString() || 0} messages
                      </div>
                      <div className="text-xs text-text-muted mt-1">
                        {identityModel.config.num_conversations || 0} conversations
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-100 border border-accent/20">
                      <div className="text-xs text-text-muted mb-1">Trained</div>
                      <div className="font-semibold text-text-primary">
                        {identityModel.config.created_at ? new Date(identityModel.config.created_at).toLocaleDateString() : 'N/A'}
                      </div>
                      <div className="text-xs text-text-muted mt-1">
                        {identityModel.config.model_size || 'Standard'} size
                      </div>
                    </div>
                  </div>
                </CollapsibleSection>
              )}


              {/* Stylistic Profile */}
              {identityModel.stylistic_profile && (
                <CollapsibleSection 
                  title="Stylistic Profile" 
                  icon={BarChart3}
                  defaultExpanded={true}
                  detailsContent={
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(identityModel.stylistic_profile).map(([key, value]: [string, any]) => {
                        let displayValue: string;
                        if (typeof value === 'number') {
                          displayValue = value.toFixed(2);
                        } else if (value && typeof value === 'object' && 'mean' in value) {
                          displayValue = `${value.mean.toFixed(2)} (σ: ${value.std.toFixed(2)})`;
                        } else {
                          displayValue = String(value);
                        }
                        
                        return (
                          <div key={key} className="p-3 rounded-lg bg-surface-100">
                            <div className="text-xs text-text-muted mb-1">{key.replace(/_/g, ' ')}</div>
                            <div className="font-medium text-text-primary text-sm">{displayValue}</div>
                          </div>
                        );
                      })}
                    </div>
                  }
                >
                  {/* Bar Chart for Stylistic Features */}
                  <div className="h-80 mb-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={Object.entries(identityModel.stylistic_profile)
                          .filter(([, value]: [string, any]) => typeof value === 'object' && 'mean' in value)
                          .map(([key, value]: [string, any]) => ({
                            feature: key.replace(/_/g, ' ').substring(0, 20),
                            mean: value.mean || 0,
                            std: value.std || 0,
                            min: value.min || 0,
                            max: value.max || 0,
                          }))
                          .slice(0, 15)
                          .sort((a, b) => b.mean - a.mean)
                        }
                        margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis 
                          dataKey="feature" 
                          angle={-45}
                          textAnchor="end"
                          height={100}
                          stroke="rgba(255,255,255,0.5)"
                          fontSize={10}
                        />
                        <YAxis stroke="rgba(255,255,255,0.5)" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                          labelStyle={{ color: '#fff' }}
                          formatter={(value: any, name: string, props: any) => {
                            if (name === 'mean') {
                              const std = props.payload?.std || 0
                              return [`${value.toFixed(4)} ± ${std.toFixed(4)}`, 'Mean ± Std Dev']
                            }
                            return [value, name]
                          }}
                        />
                        <Bar dataKey="mean" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                </CollapsibleSection>
              )}

              {/* Vocabulary Profile */}
              {identityModel.vocabulary_profile && (
                <CollapsibleSection 
                  title="Vocabulary Profile" 
                  icon={BarChart3}
                  defaultExpanded={true}
                >
                  {/* Vocabulary Category Tabs */}
                  <div className="flex gap-2 mb-4 border-b border-surface-200">
                          {identityModel.vocabulary_profile.word_frequencies && (
                            <button
                              onClick={() => setVocabTab('frequencies')}
                              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
                                vocabTab === 'frequencies'
                                  ? 'border-accent text-accent'
                                  : 'border-transparent text-text-secondary hover:text-text-primary'
                              }`}
                            >
                              Word Frequencies
                            </button>
                          )}
                          {identityModel.vocabulary_profile.distinctive_terms && Object.keys(identityModel.vocabulary_profile.distinctive_terms).length > 0 && (
                            <button
                              onClick={() => setVocabTab('distinctive')}
                              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
                                vocabTab === 'distinctive'
                                  ? 'border-accent text-accent'
                                  : 'border-transparent text-text-secondary hover:text-text-primary'
                              }`}
                            >
                              Distinctive Terms (TF-IDF)
                            </button>
                          )}
                          {identityModel.vocabulary_profile.bigrams && Object.keys(identityModel.vocabulary_profile.bigrams).length > 0 && (
                            <button
                              onClick={() => setVocabTab('bigrams')}
                              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
                                vocabTab === 'bigrams'
                                  ? 'border-accent text-accent'
                                  : 'border-transparent text-text-secondary hover:text-text-primary'
                              }`}
                            >
                              Bigrams
                            </button>
                          )}
                          {identityModel.vocabulary_profile.identity_boosted_terms && Object.keys(identityModel.vocabulary_profile.identity_boosted_terms).length > 0 && (
                            <button
                              onClick={() => setVocabTab('boosted')}
                              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${
                                vocabTab === 'boosted'
                                  ? 'border-accent text-accent'
                                  : 'border-transparent text-text-secondary hover:text-text-primary'
                              }`}
                            >
                              Identity-Boosted Terms
                            </button>
                          )}
                  </div>

                  {/* Word Cloud - Show for frequencies tab */}
                        {vocabTab === 'frequencies' && identityModel.vocabulary_profile.word_frequencies && (
                          <div className="mb-6">
                            <h4 className="text-sm font-semibold text-text-primary mb-2">Word Cloud</h4>
                            <WordCloud 
                              words={Object.entries(identityModel.vocabulary_profile.word_frequencies)} 
                              maxWords={100}
                            />
                          </div>
                        )}
                        
                  {/* Bar Chart */}
                  {vocabTab === 'frequencies' && identityModel.vocabulary_profile.word_frequencies && (
                          <div className="h-80 mb-6">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={Object.entries(identityModel.vocabulary_profile.word_frequencies)
                                  .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                                  .slice(0, 20)
                                  .map(([word, freq]: [string, any]) => ({
                                    word: word.substring(0, 15),
                                    frequency: parseFloat((freq * 100).toFixed(2)),
                                  }))}
                                margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis 
                                  dataKey="word" 
                                  angle={-45}
                                  textAnchor="end"
                                  height={100}
                                  stroke="rgba(255,255,255,0.5)"
                                  fontSize={10}
                                />
                                <YAxis 
                                  stroke="rgba(255,255,255,0.5)"
                                  label={{ value: 'Frequency (%)', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
                                />
                                <Tooltip 
                                  contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                                  labelStyle={{ color: '#fff' }}
                                  formatter={(value: any) => `${value}%`}
                                />
                                <Bar dataKey="frequency" fill="#8b5cf6" />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                  )}

                  {/* Distinctive Terms Chart */}
                  {vocabTab === 'distinctive' && identityModel.vocabulary_profile.distinctive_terms && (
                          <div className="h-80 mb-6">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={Object.entries(identityModel.vocabulary_profile.distinctive_terms)
                                  .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                                  .slice(0, 20)
                                  .map(([word, score]: [string, any]) => ({
                                    word: word.substring(0, 15),
                                    score: parseFloat(score.toFixed(4)),
                                  }))}
                                margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis 
                                  dataKey="word" 
                                  angle={-45}
                                  textAnchor="end"
                                  height={100}
                                  stroke="rgba(255,255,255,0.5)"
                                  fontSize={10}
                                />
                                <YAxis 
                                  stroke="rgba(255,255,255,0.5)"
                                  label={{ value: 'TF-IDF Score', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.7)' }}
                                />
                                <Tooltip 
                                  contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)' }}
                                  labelStyle={{ color: '#fff' }}
                                />
                                <Bar dataKey="score" fill="#a78bfa" />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                  )}

                  {/* Word Tags by Category */}
                  <div className="flex flex-wrap gap-2">
                          {vocabTab === 'frequencies' && identityModel.vocabulary_profile.word_frequencies && 
                            Object.entries(identityModel.vocabulary_profile.word_frequencies)
                              .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                              .slice(0, 50)
                              .map(([word, freq]: [string, any]) => (
                                <div 
                                  key={word} 
                                  className="px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium"
                                  title={`Frequency: ${(freq * 100).toFixed(2)}%`}
                                >
                                  {word}
                                </div>
                              ))
                          }
                          {vocabTab === 'distinctive' && identityModel.vocabulary_profile.distinctive_terms &&
                            Object.entries(identityModel.vocabulary_profile.distinctive_terms)
                              .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                              .slice(0, 50)
                              .map(([word, score]: [string, any]) => (
                                <div 
                                  key={word} 
                                  className="px-3 py-1 rounded-full bg-purple-500/10 text-purple-400 text-sm font-medium"
                                  title={`TF-IDF Score: ${score.toFixed(3)}`}
                                >
                                  {word}
                                </div>
                              ))
                          }
                          {vocabTab === 'bigrams' && identityModel.vocabulary_profile.bigrams &&
                            Object.entries(identityModel.vocabulary_profile.bigrams)
                              .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                              .slice(0, 50)
                              .map(([bigram, count]: [string, any]) => (
                                <div 
                                  key={bigram} 
                                  className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-sm font-medium"
                                  title={`Count: ${count}`}
                                >
                                  {bigram.replace('_', ' ')}
                                </div>
                              ))
                          }
                          {vocabTab === 'boosted' && identityModel.vocabulary_profile.identity_boosted_terms &&
                            Object.entries(identityModel.vocabulary_profile.identity_boosted_terms)
                              .sort(([, a]: [string, any], [, b]: [string, any]) => (b || 0) - (a || 0))
                              .slice(0, 50)
                              .map(([word, boost]: [string, any]) => (
                                <div 
                                  key={word} 
                                  className="px-3 py-1 rounded-full bg-green-500/10 text-green-400 text-sm font-medium"
                                  title={`Boost Factor: ${boost.toFixed(2)}x`}
                                >
                                  {word}
                                </div>
                              ))
                          }
                  </div>
                  {identityModel.vocabulary_profile.vocabulary_size && (
                    <div className="mt-4 text-sm text-text-muted">
                      Vocabulary size: {identityModel.vocabulary_profile.vocabulary_size.toLocaleString()} words, 
                      Total: {identityModel.vocabulary_profile.total_words?.toLocaleString() || 'N/A'} words
                      {vocabTab === 'bigrams' && identityModel.vocabulary_profile.bigrams && (
                        <span className="ml-2">
                          • {Object.keys(identityModel.vocabulary_profile.bigrams).length} bigrams
                        </span>
                      )}
                      {vocabTab === 'distinctive' && identityModel.vocabulary_profile.distinctive_terms && (
                        <span className="ml-2">
                          • {Object.keys(identityModel.vocabulary_profile.distinctive_terms).length} distinctive terms
                        </span>
                      )}
                      {vocabTab === 'boosted' && identityModel.vocabulary_profile.identity_boosted_terms && (
                        <span className="ml-2">
                          • {Object.keys(identityModel.vocabulary_profile.identity_boosted_terms).length} boosted terms
                        </span>
                      )}
                    </div>
                  )}
                </CollapsibleSection>
              )}

              {/* Identity Report Summary - Moved to end after visualizations */}
              {identityModel.identity_report && (
                <CollapsibleSection 
                  title="Identity Analysis Report" 
                  icon={FileText}
                  defaultExpanded={false}
                >
                  <div className="prose prose-sm max-w-none text-text-secondary bg-surface-100 p-4 rounded-lg overflow-auto max-h-96">
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {identityModel.identity_report.split('\n').map((line: string, idx: number) => {
                        // Format markdown-like headers
                        if (line.startsWith('# ')) {
                          return <h1 key={idx} className="text-lg font-bold text-text-primary mt-4 mb-2">{line.substring(2)}</h1>
                        }
                        if (line.startsWith('## ')) {
                          return <h2 key={idx} className="text-base font-semibold text-text-primary mt-3 mb-2">{line.substring(3)}</h2>
                        }
                        if (line.startsWith('### ')) {
                          return <h3 key={idx} className="text-sm font-semibold text-text-primary mt-2 mb-1">{line.substring(4)}</h3>
                        }
                        if (line.startsWith('- ') || line.startsWith('* ')) {
                          return <div key={idx} className="ml-4 text-text-secondary">• {line.substring(2)}</div>
                        }
                        if (line.trim() === '---') {
                          return <hr key={idx} className="my-4 border-surface-200" />
                        }
                        return <div key={idx} className="text-text-secondary mb-1">{line || '\u00A0'}</div>
                      })}
                    </div>
                  </div>
                  <button 
                    onClick={() => {
                      setSelectedItem({ type: 'report', title: 'Identity Analysis Report' })
                      setEditorContent(identityModel.identity_report || '')
                    }}
                    className="btn btn-secondary mt-4"
                  >
                    <Eye className="w-4 h-4" />
                    View Full Report in Editor
                  </button>
                </CollapsibleSection>
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
              <div className="flex-1 max-w-md ml-4">
                <SearchInput
                  placeholder="Search files..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {loading ? (
              <div className="text-center py-12 text-text-muted">Loading...</div>
            ) : files.length === 0 ? (
              <EmptyState
                icon={FolderOpen}
                title="No files uploaded yet"
                message="Upload files to get started"
              />
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

