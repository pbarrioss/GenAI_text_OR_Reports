# Minimal Flask App - Save as minimal_app.py
from flask import Flask, request, jsonify
import torch
from transformers import BioGptForCausalLM, BioGptTokenizer

app = Flask(__name__)

# Load model 
print("Loading model...")
model = BioGptForCausalLM.from_pretrained('./operative-report-model')
tokenizer = BioGptTokenizer.from_pretrained('./operative-report-model')
print("Model loaded!")

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Operative Report Generator</title></head>
    <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
        <div style="max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h1 style="color: #2c3e50; text-align: center;">🏥 Operative Report Generator</h1>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h3>Input & Controls</h3>
                    
                    <p><label><strong>Clinical Indication:</strong></label></p>
                    <textarea id="brief" rows="3" style="width: 100%; padding: 8px;" placeholder="e.g., Laparoscopic appendectomy for acute appendicitis"></textarea>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                        <div>
                            <label><strong>Temperature:</strong> <span id="tempDisplay">0.8</span></label>
                            <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.8" style="width: 100%;" 
                                oninput="document.getElementById('tempDisplay').innerText=this.value">
                            <small style="color: #666;">Higher = more creative</small>
                        </div>
                        
                        <div>
                            <label><strong>Max Length:</strong> <span id="lengthDisplay">300</span></label>
                            <input type="range" id="length" min="100" max="600" step="50" value="300" style="width: 100%;" 
                                oninput="document.getElementById('lengthDisplay').innerText=this.value">
                            <small style="color: #666;">Tokens to generate</small>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                        <div>
                            <label><strong>Top-p:</strong> <span id="toppDisplay">0.9</span></label>
                            <input type="range" id="topp" min="0.1" max="1.0" step="0.1" value="0.9" style="width: 100%;" 
                                oninput="document.getElementById('toppDisplay').innerText=this.value">
                            <small style="color: #666;">Nucleus sampling</small>
                        </div>
                        
                        <div>
                            <label><strong>Top-k:</strong> <span id="topkDisplay">50</span></label>
                            <input type="range" id="topk" min="10" max="100" step="10" value="50" style="width: 100%;" 
                                oninput="document.getElementById('topkDisplay').innerText=this.value">
                            <small style="color: #666;">Word choice limit</small>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                        <div>
                            <label><strong>Repetition Penalty:</strong> <span id="repDisplay">1.2</span></label>
                            <input type="range" id="rep" min="1.0" max="2.0" step="0.1" value="1.2" style="width: 100%;" 
                                oninput="document.getElementById('repDisplay').innerText=this.value">
                            <small style="color: #666;">Reduce repetition</small>
                        </div>
                        
                        <div>
                            <label><strong>No Repeat N-gram:</strong> <span id="ngramDisplay">3</span></label>
                            <input type="range" id="ngram" min="2" max="5" step="1" value="3" style="width: 100%;" 
                                oninput="document.getElementById('ngramDisplay').innerText=this.value">
                            <small style="color: #666;">Block repeated phrases</small>
                        </div>
                    </div>
                    
                    <div style="margin: 15px 0;">
                        <label><strong>Sampling Strategy:</strong></label><br>
                        <input type="radio" id="sample" name="strategy" value="sample" checked> 
                        <label for="sample">Sampling (creative)</label><br>
                        <input type="radio" id="greedy" name="strategy" value="greedy"> 
                        <label for="greedy">Greedy (conservative)</label><br>
                        <input type="radio" id="beam" name="strategy" value="beam"> 
                        <label for="beam">Beam Search (balanced)</label>
                    </div>
                    
                    <button onclick="generate()" style="width: 100%; background: #007bff; color: white; border: none; padding: 12px; border-radius: 5px; font-size: 16px; cursor: pointer;">
                        Generate Report
                    </button>
                    
                    <div style="margin-top: 15px;">
                        <button onclick="setPreset('conservative')" style="background: #28a745; color: white; border: none; padding: 8px 12px; margin: 2px; border-radius: 4px; cursor: pointer;">Conservative</button>
                        <button onclick="setPreset('balanced')" style="background: #17a2b8; color: white; border: none; padding: 8px 12px; margin: 2px; border-radius: 4px; cursor: pointer;">Balanced</button>
                        <button onclick="setPreset('creative')" style="background: #ffc107; color: black; border: none; padding: 8px 12px; margin: 2px; border-radius: 4px; cursor: pointer;">Creative</button>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h3>Generated Report</h3>
                    <div id="loading" style="display:none; color: #007bff; font-style: italic; margin: 10px 0;">🔄 Generating report...</div>
                    <textarea id="result" rows="20" style="width: 100%; padding: 8px;" readonly placeholder="Generated operative report will appear here..."></textarea>
                    
                    <div style="margin-top: 10px;">
                        <button onclick="copyReport()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Copy Report</button>
                        <button onclick="clearReport()" style="background: #dc3545; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-left: 5px;">Clear</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function setPreset(type) {
            if (type === 'conservative') {
                document.getElementById('temp').value = 0.3;
                document.getElementById('topp').value = 0.7;
                document.getElementById('topk').value = 30;
                document.getElementById('rep').value = 1.3;
                document.getElementById('greedy').checked = true;
            } else if (type === 'balanced') {
                document.getElementById('temp').value = 0.8;
                document.getElementById('topp').value = 0.9;
                document.getElementById('topk').value = 50;
                document.getElementById('rep').value = 1.2;
                document.getElementById('sample').checked = true;
            } else if (type === 'creative') {
                document.getElementById('temp').value = 1.2;
                document.getElementById('topp').value = 0.95;
                document.getElementById('topk').value = 80;
                document.getElementById('rep').value = 1.1;
                document.getElementById('sample').checked = true;
            }
            updateDisplays();
        }
        
        function updateDisplays() {
            document.getElementById('tempDisplay').innerText = document.getElementById('temp').value;
            document.getElementById('lengthDisplay').innerText = document.getElementById('length').value;
            document.getElementById('toppDisplay').innerText = document.getElementById('topp').value;
            document.getElementById('topkDisplay').innerText = document.getElementById('topk').value;
            document.getElementById('repDisplay').innerText = document.getElementById('rep').value;
            document.getElementById('ngramDisplay').innerText = document.getElementById('ngram').value;
        }
        
        function copyReport() {
            document.getElementById('result').select();
            document.execCommand('copy');
            alert('Report copied to clipboard!');
        }
        
        function clearReport() {
            document.getElementById('result').value = '';
        }
        
        async function generate() {
            const brief = document.getElementById('brief').value;
            const params = {
                brief: brief,
                temperature: parseFloat(document.getElementById('temp').value),
                max_length: parseInt(document.getElementById('length').value),
                top_p: parseFloat(document.getElementById('topp').value),
                top_k: parseInt(document.getElementById('topk').value),
                repetition_penalty: parseFloat(document.getElementById('rep').value),
                no_repeat_ngram_size: parseInt(document.getElementById('ngram').value),
                strategy: document.querySelector('input[name="strategy"]:checked').value
            };
            
            if (!brief.trim()) {
                alert('Please enter a clinical indication');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').value = '';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').value = data.report || 'Error: ' + data.error;
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').value = 'Error: ' + error.message;
            }
        }
        </script>
    </body>
    </html>
    '''

@app.route('/api/generate', methods=['POST'])
def generate_api():
    try:
        data = request.json
        brief = data.get('brief', '').strip()
        
        # Get all parameters with defaults
        temperature = data.get('temperature', 0.8)
        max_length = data.get('max_length', 300)
        top_p = data.get('top_p', 0.9)
        top_k = data.get('top_k', 50)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        no_repeat_ngram_size = data.get('no_repeat_ngram_size', 3)
        strategy = data.get('strategy', 'sample')
        
        if not brief:
            return jsonify({'error': 'No clinical indication provided'})
        
        # Generate report with medical format (matching test code)
        prompt = f"PROCEDURE: Appendectomy\nINDICATION: {brief}\nOPERATIVE REPORT:"
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones_like(inputs)
        
        # Configure generation parameters based on strategy
        gen_kwargs = {
            'input_ids': inputs,
            'attention_mask': attention_mask,
            'max_length': len(inputs[0]) + max_length,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'repetition_penalty': repetition_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'early_stopping': True,
            
            'bad_words_ids': [
                tokenizer.encode("<", add_special_tokens=False),
                tokenizer.encode(">", add_special_tokens=False),
                tokenizer.encode("endoftext", add_special_tokens=False),
                tokenizer.encode("AbstractText", add_special_tokens=False),
                tokenizer.encode("NlmCategory", add_special_tokens=False),
                tokenizer.encode("UNASSIGNED", add_special_tokens=False),
                tokenizer.encode("ns0:", add_special_tokens=False),
                tokenizer.encode("mml:", add_special_tokens=False)
            ]
        }
        
        if strategy == 'greedy':
            gen_kwargs.update({
                'do_sample': False
            })
        elif strategy == 'beam':
            gen_kwargs.update({
                'num_beams': 3,
                'do_sample': False
            })
        else:  # sampling
            gen_kwargs.update({
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            })
        
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        garbage_tokens = ["<", "endoftext", "AbstractText", "NlmCategory", "UNASSIGNED", "ns0:", "mml:", "≥", "≤"]
        if any(token in full_text for token in garbage_tokens):
            # Find first garbage token and cut there
            cut_points = []
            for token in garbage_tokens:
                if token in full_text:
                    cut_points.append(full_text.find(token))
            if cut_points:
                full_text = full_text[:min(cut_points)]
        
        # Extract just the report part (updated for medical format)
        if "OPERATIVE REPORT:" in full_text:
            report = full_text.split("OPERATIVE REPORT:")[1].strip()
        else:
            report = full_text
        
        # Advanced cleanup - match test script cleaning logic
        clean_report = _clean_repetitive_text(report)
            
        return jsonify({'report': clean_report})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def _clean_repetitive_text(text):
    """Remove repetitive sentences - same logic as test script"""
    sentences = text.split('.')
    cleaned = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            is_repetitive = False
            for prev in cleaned[-2:]:
                if sentence in prev or prev in sentence:
                    is_repetitive = True
                    break
            
            if not is_repetitive:
                cleaned.append(sentence)
            else:
                break
    
    return '. '.join(cleaned) + '.' if cleaned else text

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Using port 5001 to avoid conflicts