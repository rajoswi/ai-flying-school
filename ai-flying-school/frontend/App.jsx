
import React, {useState} from 'react'

export default function App(){
  const [rows,setRows] = useState([])
  const [loading,setLoading] = useState(false)

  async function uploadCSV(e){
    const file = e.target.files[0]
    if(!file) return
    setLoading(true)
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('http://localhost:8000/predict_csv', {method:'POST',body:fd})
    const data = await res.json()
    setRows(data)
    setLoading(false)
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">AI Cadet Risk Dashboard</h1>
      <div className="mb-4">
        <input type="file" accept=".csv" onChange={uploadCSV} />
      </div>
      {loading && <div>Predicting...</div>}
      <div className="grid grid-cols-1 gap-3">
        {rows.map((r,idx)=> (
          <div key={idx} className={`p-4 rounded shadow ${r.risk_label==='High'? 'border-l-4 border-red-500':''} ${r.risk_label==='Medium'? 'border-l-4 border-yellow-400':''}`}>
            <div className="flex justify-between">
              <div>
                <div className="font-semibold">{r.name} — {r.cadet_id}</div>
                <div className="text-sm text-gray-600">Batch: {r.batch}</div>
              </div>
              <div className="text-right">
                <div className="text-sm">Risk: <span className={`font-bold ${r.risk_label==='High' ? 'text-red-600' : r.risk_label==='Medium' ? 'text-yellow-600' : 'text-green-600'}`}>{r.risk_label}</span></div>
                <div className="text-xs text-gray-500">Score: {parseFloat(r.risk_score).toFixed(2)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
