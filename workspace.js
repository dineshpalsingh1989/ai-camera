/* Monitoring tools and the reviewed-data training workspace. Uses the app's authenticated API. */
(() => {
  let training = null, loading = false, firstLoad = true, photoPage = 0, editing = null, boxes = [], drag = null;
  let selected = new Set(), focusCamera = null, reviewEvent = null, photoVersion = '';
  const labelImage = new Image(), labelCanvas = $('#label-canvas'), labelContext = labelCanvas.getContext('2d');
  const endpoint = '/api/training';
  const percent = value => value == null ? '—' : (value * 100).toFixed(1) + '%';
  const stamp = value => value ? new Date(value * 1000).toLocaleString() : 'Not yet';
  const post = (path, json = {}) => api(endpoint + path, { json });
  const message = (text, bad = false) => {
    const el = $('#training-message'); el.textContent = text; el.classList.remove('hidden'); el.classList.toggle('error', bad);
  };
  const action = (selector, callback) => {
    $(selector).onclick = async e => {
      const button = e.currentTarget; button.disabled = true;
      try { await callback(); await loadTraining(true); }
      catch (err) { message(err.message, true); }
      finally { button.disabled = false; }
    };
  };
  function options(selector, rows, emptyLabel) {
    const el = $(selector), old = el.value;
    const html = (emptyLabel != null ? `<option value="">${esc(emptyLabel)}</option>` : '') + rows.map(r => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');
    if (el.innerHTML !== html) { el.innerHTML = html; if ([...el.options].some(o => o.value === old)) el.value = old; }
  }

  // Camera groups and focus keep existing tile elements, streams and controls alive.
  window.decorateCameras = () => {
    if (!S) return;
    const group = $('#camera-group').value;
    const groups = [...new Set(S.cameras.map(c => c.group).filter(Boolean))].sort();
    $('#camera-group').innerHTML = '<option value="">All groups</option>' + groups.map(g => `<option>${esc(g)}</option>`).join('');
    if (groups.includes(group)) $('#camera-group').value = group;
    S.cameras.filter(c => c.enabled).forEach(c => {
      const tile = $('#t-' + c.id); if (!tile || tile.querySelector('[data-focus]')) return;
      const button = document.createElement('button'); button.className = 'btn'; button.dataset.focus = c.id;
      button.textContent = 'Focus'; button.title = 'Focus on this camera while keeping other feeds visible';
      button.onclick = () => setFocus(focusCamera === c.id ? null : c.id);
      tile.querySelector('.tile-actions').append(button);
    });
    focusCamera = null; setFocus(null); filterCameras();
  };
  function setFocus(id) {
    focusCamera = id; $('#grid').classList.toggle('focus-mode', !!id);
    $$('#grid .tile').forEach(t => {
      const active = t.id === 't-' + id; t.classList.toggle('focused-camera', active);
      const button = t.querySelector('[data-focus]'); if (button) { button.textContent = active ? 'Unfocus' : 'Focus'; button.setAttribute('aria-pressed', String(active)); }
    });
    if (id) $('#grid').prepend($('#t-' + id));
    else if (S) S.cameras.forEach(c => { const tile = $('#t-' + c.id); if (tile) $('#grid').append(tile); });
    $('#exit-focus').classList.toggle('hidden', !id);
  }
  function filterCameras() {
    if (!S) return;
    const query = $('#camera-search').value.trim().toLowerCase(), group = $('#camera-group').value;
    let count = 0;
    S.cameras.filter(c => c.enabled).forEach(c => {
      const tile = $('#t-' + c.id); if (!tile) return;
      const match = (!group || c.group === group) && [c.name,c.location,c.group].join(' ').toLowerCase().includes(query);
      tile.classList.toggle('hidden', !match); if (match) count++;
      if (!match && focusCamera === c.id) setFocus(null);
    });
    $('#camera-filter-empty').classList.toggle('hidden', count > 0 || !S.cameras.some(c => c.enabled));
    $('#camera-count').textContent = count;
  }
  $('#camera-search').oninput = $('#camera-group').onchange = filterCameras;
  $('#exit-focus').onclick = () => setFocus(null);
  window.updateWorkspaceStatus = status => {
    if (!S) return;  // no settings means nothing to show; a missing status just leaves the live columns blank
    const byId = Object.fromEntries((status?.cameras || []).map(c => [c.id,c]));
    $('#health-rows').innerHTML = S.cameras.length ? S.cameras.map(c => {
      const live = byId[c.id], age = live?.last_frame ? Date.now()/1000 - live.last_frame : null;
      const healthy = !!live?.online && age != null && age < 15;
      const state = !c.enabled ? 'Disabled' : healthy ? 'Connected' : live?.online ? 'Stale frames' : 'Disconnected';
      return `<tr><td>${esc(c.name)}</td><td>${esc(c.group || 'Ungrouped')}</td><td><span class="pill ${healthy ? 'ok' : c.enabled ? 'bad' : ''}">${state}</span></td><td>${live ? live.fps : '—'}</td><td>${esc(stamp(live?.last_frame))}</td><td>${esc(stamp(live?.last_connected))}</td><td>${live?.reconnects ?? '—'}</td></tr>`;
    }).join('') : '<tr><td colspan="7">Add a camera in Settings to see its connection health.</td></tr>';
  };

  // Alert review stays separate from training labels: a review is not a bounding-box annotation.
  window.openEventReview = event => {
    reviewEvent = event;
    $('#event-review').classList.toggle('hidden', !event.id);
    $('#event-outcome').value = event.outcome || (event.reviewed ? 'confirmed' : 'unreviewed');
    $('#event-note').value = event.review_note || ''; $('#event-review-status').textContent = '';
  };
  $('#save-event-review').onclick = async e => {
    if (!reviewEvent?.id) return;
    const button = e.currentTarget; button.disabled = true;
    try {
      await api('/api/events/' + encodeURIComponent(reviewEvent.id) + '/review', {json:{outcome:$('#event-outcome').value,note:$('#event-note').value}});
      $('#event-review-status').textContent = 'Review saved';
      await Promise.all([loadEvents(), pollFeed()]);
    } catch (err) { $('#event-review-status').textContent = err.message; }
    finally { button.disabled = false; }
  };
  ['#f-from','#f-to','#f-severity','#f-review'].forEach(id => $(id).onchange = () => loadEvents().catch(err => alert(err.message)));

  $$('.settings-tabs button').forEach(button => button.onclick = () => {
    $$('.settings-tabs button').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-selected', String(b === button)); });
    $$('[data-settings-pane]').forEach(p => p.classList.toggle('hidden', p.dataset.settingsPane !== button.dataset.settings));
  });
  $$('.training-tabs button').forEach(button => button.onclick = () => {
    $$('.training-tabs button').forEach(b => b.classList.toggle('active', b === button));
    $$('[data-training-pane]').forEach(p => p.classList.toggle('hidden', p.dataset.trainingPane !== button.dataset.training));
  });

  async function loadTraining(force = false) {
    if (loading) return;
    loading = true;
    try {
      training = await api(endpoint);
      renderTraining(force);
    } catch (err) { message(err.message, true); }
    finally { loading = false; }
  }
  window.loadTraining = loadTraining;
  function renderTraining(force) {
    const cameras = S?.cameras || [];
    options('#learn-camera', cameras, cameras.length ? null : 'No cameras configured');
    const datasets = training.datasets.map(d => ({id:d.id,name:`${d.name} · ${d.counts.train || 0} train / ${d.counts.val || 0} validation`}));
    options('#train-dataset', datasets, datasets.length ? null : 'Build or import a dataset first');
    options('#evaluate-dataset', datasets, datasets.length ? null : 'Choose a dataset');
    options('#train-base', training.models, 'YOLOv8 nano (installed base model)');
    const chosen = new Set($$('#activation-cameras input:checked').map(i=>i.value));
    const before = $('#activation-cameras').dataset.cameras;
    const key = JSON.stringify(cameras.map(c=>[c.id,c.name]));
    if (before !== key) {
      $('#activation-cameras').dataset.cameras = key;
      $('#activation-cameras').innerHTML = cameras.map(c => `<label class="inline"><input type="checkbox" value="${esc(c.id)}" ${chosen.has(c.id) ? 'checked' : ''}>${esc(c.name)}</label>`).join('') || '<small>Add cameras in Settings first.</small>';
    }
    if (firstLoad) {
      $('#rf-workspace').value = training.connection.workspace; $('#rf-project').value = training.connection.project;
      syncAutoRule(); firstLoad = false;
    }
    $('#rf-key-status').textContent = training.connection.key_set ? '(saved)' : '(not set)';
    $('#rf-open').classList.toggle('hidden', !training.connection.workspace || !training.connection.project);
    $('#rf-open').href = `https://app.roboflow.com/${encodeURIComponent(training.connection.workspace)}/${encodeURIComponent(training.connection.project)}`;
    $('#rollback-model').disabled = !training.can_rollback;
    $('#active-model-label').textContent = 'Active custom model: ' + training.active_model;
    $('#photo-summary').textContent = `${training.photos.length} photos · ${training.photos.filter(p=>p.reviewed).length} reviewed`;
    $('#known-labels').innerHTML = [...new Set([...(S?.custom_classes||[]), ...training.photos.flatMap(p=>p.boxes.map(b=>b.label))])].map(n=>`<option value="${esc(n)}">`).join('');
    const version = JSON.stringify(training.photos.map(p=>[p.id,p.reviewed,p.split,p.boxes,p.uploads]));
    if (force || photoVersion !== version) { photoVersion = version; renderPhotos(); }
    setHTML($('#dataset-list'), training.datasets.map(d=>`<div class="dataset-row"><div><b>${esc(d.name)}</b><small>${esc(d.classes.join(', '))}<br>${d.counts.train||0} train · ${d.counts.val||0} validation · ${d.counts.test||0} test</small></div><a class="btn" href="${endpoint}/datasets/${d.id}/download">Download ZIP</a><button class="btn danger" data-delete-dataset="${d.id}">Delete</button></div>`).join('') || '<p class="hint">No datasets yet. Review photos or import a labeled YOLO dataset.</p>');
    $$('[data-delete-dataset]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete this dataset snapshot? Trained models keep working; this only removes the snapshot.'))return;b.disabled=true;try{await api(endpoint+'/datasets/'+b.dataset.deleteDataset,{method:'DELETE'});message('Dataset deleted.');await loadTraining();}catch(err){message(err.message,true);b.disabled=false;}});
    renderModels(); renderJobs();
  }
  function visiblePhotos() {
    if (!training) return [];
    const filter = $('#photo-filter').value;
    return training.photos.filter(p => !filter || p.reviewed === (filter === 'reviewed'));
  }
  function renderPhotos() {
    const all = visiblePhotos(), pages = Math.max(1,Math.ceil(all.length/24)); photoPage = Math.min(photoPage,pages-1);
    const photos = all.slice(photoPage*24,photoPage*24+24);
    const exists = new Set(training.photos.map(p=>p.id)); selected = new Set([...selected].filter(id=>exists.has(id)));
    $('#photo-grid').innerHTML = photos.map(p=>`<article class="photo-card"><label class="photo-select"><input type="checkbox" data-photo-select="${p.id}" aria-label="Select ${esc(p.source)}" ${selected.has(p.id)?'checked':''}></label><button class="photo-open" data-photo-open="${p.id}"><img loading="lazy" src="${endpoint}/photos/${p.id}/image" alt="${esc(p.source)}"><span class="photo-copy"><b>${esc(p.source)}</b><small>${p.boxes.length} boxes · ${p.split === 'val' ? 'validation' : p.split}</small><span class="chip ${p.reviewed?'reviewed':''}">${p.reviewed?'Reviewed':'Needs review'}</span></span></button></article>`).join('') || emptyState('photo','Start with real examples','Capture a camera frame, upload images, or import photos collected in camera settings.');
    $$('[data-photo-select]').forEach(input => input.onchange = () => { input.checked ? selected.add(input.dataset.photoSelect) : selected.delete(input.dataset.photoSelect); selectionCount(); });
    $$('[data-photo-open]').forEach(button => button.onclick = () => openLabelEditor(button.dataset.photoOpen));
    $('#photos-page').textContent = `Page ${photoPage+1} of ${pages}`; $('#photos-prev').disabled = !photoPage; $('#photos-next').disabled = photoPage >= pages-1;
    selectionCount();
  }
  function selectionCount() { $('#selected-count').textContent = `${selected.size} selected`; }
  $('#photos-prev').onclick = () => { photoPage--; renderPhotos(); };
  $('#photos-next').onclick = () => { photoPage++; renderPhotos(); };
  $('#photo-filter').onchange = () => { photoPage=0; renderPhotos(); };
  $('#select-photos').onclick = () => {
    const inputs = $$('[data-photo-select]'), all = inputs.every(i=>i.checked);
    inputs.forEach(i=>{ i.checked=!all; all?selected.delete(i.dataset.photoSelect):selected.add(i.dataset.photoSelect); }); selectionCount();
  };
  action('#refresh-training', async () => { message('Workspace refreshed'); });
  action('#capture-photo', async () => { const camera=$('#learn-camera').value; if(!camera)return message('Select a camera first — add one in Settings → Cameras.',true); await post('/capture',{camera,split:$('#capture-split').value}); message('Photo captured. Review its labels before training.'); });
  action('#import-collected', async () => { const result=await post('/import-collected',{camera:$('#learn-camera').value,split:$('#capture-split').value}); message(`Imported ${result.imported} photos; exact duplicates are skipped.`); });
  function uploadFile(selector, path, field, extra = () => ({})) {
    $(selector).onchange = async e => {
      const input=e.currentTarget; if (!input.files.length) return;
      input.disabled=true;
      try {
        const data=new FormData(); [...input.files].forEach(f=>data.append(field,f)); Object.entries(extra()).forEach(([k,v])=>data.append(k,v));
        message('Uploading…'); await api(endpoint+path,{method:'POST',body:data}); message('Upload received. Check the library or job status below.'); await loadTraining(true);
      } catch(err) { message(err.message,true); }
      finally { input.value=''; input.disabled=false; }
    };
  }
  uploadFile('#upload-photos','/photos','files',()=>({split:$('#capture-split').value}));
  uploadFile('#import-dataset','/datasets','file'); uploadFile('#import-model','/models','file');
  action('#upload-selected', async () => {
    if (!selected.size) throw new Error('Select photos to upload first');
    if (!training.connection.key_set) { $('[data-training="roboflow"]').click(); throw new Error('Connect your Roboflow project first'); }
    await post('/roboflow/upload',{photos:[...selected]}); message('Roboflow upload started. Progress is shown below.');
  });
  function syncAutoRule() {
    const rule=training?.auto_collect[$('#learn-camera').value];
    $('#auto-enabled').checked=!!rule?.enabled; $('#auto-interval').value=rule?.interval||30; $('#auto-confidence').value=rule?.confidence||.7;
  }
  $('#learn-camera').onchange=syncAutoRule;
  action('#save-auto',async()=>{
    await post('/auto-collect',{camera:$('#learn-camera').value,enabled:$('#auto-enabled').checked,interval:+$('#auto-interval').value,confidence:+$('#auto-confidence').value,split:$('#capture-split').value});
    message('Collection rule saved. New examples require review before training.');
  });

  // Label editor stores normalized boxes so annotations do not depend on display size.
  function openLabelEditor(id) {
    const photo=training.photos.find(p=>p.id===id); if(!photo)return;
    if(editing && !confirm('Open another photo? Any unsaved label edits will be discarded.'))return;
    editing=id; boxes=structuredClone(photo.boxes); $('#label-split').value=photo.split; $('#label-reviewed').checked=photo.reviewed;
    labelImage.src=`${endpoint}/photos/${id}/image`; $('#label-editor').classList.remove('hidden');
    $('#label-editor').scrollIntoView({behavior:'smooth',block:'start'}); renderBoxList();
  }
  labelImage.onload=()=>{labelCanvas.width=Math.min(labelImage.naturalWidth,1280); labelCanvas.height=Math.round(labelCanvas.width*labelImage.naturalHeight/labelImage.naturalWidth); drawLabels();};
  function drawLabels() {
    labelContext.clearRect(0,0,labelCanvas.width,labelCanvas.height);
    if(labelImage.complete && labelImage.naturalWidth)labelContext.drawImage(labelImage,0,0,labelCanvas.width,labelCanvas.height);
    [...boxes,...(drag?[{...drag,label:$('#label-class').value||'Object'}]:[])].forEach((b,i)=>{
      const x=b.x1*labelCanvas.width,y=b.y1*labelCanvas.height,w=(b.x2-b.x1)*labelCanvas.width,h=(b.y2-b.y1)*labelCanvas.height;
      labelContext.strokeStyle='#57dfbf';labelContext.lineWidth=2;labelContext.strokeRect(x,y,w,h);
      labelContext.font='15px system-ui';const text=`${i+1}. ${b.label}`;const width=labelContext.measureText(text).width+8;
      labelContext.fillStyle='#0c211e';labelContext.fillRect(x,Math.max(0,y-22),width,22);labelContext.fillStyle='#95f5da';labelContext.fillText(text,x+4,Math.max(16,y-6));
    });
  }
  function pointer(event){const rect=labelCanvas.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))};}
  labelCanvas.onpointerdown=e=>{if(!$('#label-class').value.trim()){message('Enter an object class before drawing.',true);$('#label-class').focus();return;}const p=pointer(e);drag={x1:p.x,y1:p.y,x2:p.x,y2:p.y};labelCanvas.setPointerCapture(e.pointerId);};
  labelCanvas.onpointermove=e=>{if(!drag)return;const p=pointer(e);drag.x2=p.x;drag.y2=p.y;drawLabels();};
  labelCanvas.onpointerup=()=>{if(!drag)return;const b={label:$('#label-class').value.trim(),x1:Math.min(drag.x1,drag.x2),y1:Math.min(drag.y1,drag.y2),x2:Math.max(drag.x1,drag.x2),y2:Math.max(drag.y1,drag.y2)};drag=null;if(b.x2-b.x1>.002&&b.y2-b.y1>.002)boxes.push(b);$('#label-reviewed').checked=false;renderBoxList();drawLabels();};
  labelCanvas.onpointercancel=()=>{drag=null;drawLabels();};
  function renderBoxList(){
    $('#label-box-list').innerHTML=boxes.map((b,i)=>`<div class="label-row"><span>${i+1}. ${esc(b.label)}</span><button class="btn" data-remove-box="${i}" aria-label="Remove box ${i+1}">Remove</button></div>`).join('')||'<p class="hint">No boxes. Mark reviewed only if there are no target objects in this image.</p>';
    $$('[data-remove-box]').forEach(b=>b.onclick=()=>{boxes.splice(+b.dataset.removeBox,1);$('#label-reviewed').checked=false;renderBoxList();drawLabels();});
  }
  $('#add-label-box').onclick=()=>{
    const b={label:$('#label-class').value.trim(),x1:+$('#box-x1').value/100,y1:+$('#box-y1').value/100,x2:+$('#box-x2').value/100,y2:+$('#box-y2').value/100};
    if(!b.label||b.x2<=b.x1||b.y2<=b.y1||[b.x1,b.y1,b.x2,b.y2].some(n=>!Number.isFinite(n)||n<0||n>1))return message('Enter a class and a valid rectangle between 0 and 100%.',true);
    boxes.push(b);$('#label-reviewed').checked=false;renderBoxList();drawLabels();
  };
  $('#undo-label-box').onclick=()=>{boxes.pop();$('#label-reviewed').checked=false;renderBoxList();drawLabels();};
  $('#clear-label-boxes').onclick=()=>{boxes=[];$('#label-reviewed').checked=false;renderBoxList();drawLabels();};
  $('#close-label-editor').onclick=()=>{editing=null;$('#label-editor').classList.add('hidden');};
  action('#save-labels',async()=>{
    if(!editing)return;
    await api(endpoint+'/photos/'+editing,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({boxes,split:$('#label-split').value,reviewed:$('#label-reviewed').checked})});
    message('Photo labels and review saved.');
  });
  action('#delete-photo',async()=>{if(!editing||!confirm('Delete this training photo and its labels? Existing dataset snapshots are kept.'))return;await api(endpoint+'/photos/'+editing,{method:'DELETE'});editing=null;$('#label-editor').classList.add('hidden');message('Photo deleted.');});
  action('#build-dataset',async()=>{const d=await post('/datasets');message(`Dataset built: ${d.counts.train} training and ${d.counts.val} validation images.`);});
  action('#auto-split',async()=>{const r=await post('/auto-split');message(`Auto-split done: ${r.train} training and ${r.val} validation. Now click "Build from reviewed photos".`);});
  action('#start-training',async()=>{
    await post('/train',{dataset:$('#train-dataset').value,name:$('#train-name').value,base_model:$('#train-base').value,epochs:+$('#train-epochs').value,imgsz:+$('#train-size').value,batch:+$('#train-batch').value,device:$('#train-device').value});
    message('Training started. You can monitor cameras while the job runs.');
  });
  function renderModels(){
    const did=$('#evaluate-dataset').value;
    const html=training.models.map(m=>{
      const metrics=m.evaluations[did], active=training.active_model.replaceAll('\\','/').endsWith('/'+m.file.replaceAll('\\','/'));
      return `<article class="model-card"><div class="row"><h3>${esc(m.name)}</h3><span class="chip ${active?'reviewed':''}">${active?'Active':'Candidate'}</span></div><p class="hint">${esc(m.classes.join(', '))}</p><div class="model-metrics"><span>Precision<b>${percent(metrics?.precision)}</b></span><span>Recall<b>${percent(metrics?.recall)}</b></span><span>mAP@50<b>${percent(metrics?.map50)}</b></span><span>mAP@50–95<b>${percent(metrics?.map50_95)}</b></span></div><p class="hint">${metrics?'Scores on the selected comparison dataset.':'Not evaluated on the selected dataset.'}</p><div class="row"><button class="btn" data-evaluate="${m.id}" ${!did?'disabled':''}>Evaluate</button><button class="btn ${active?'':'primary'}" data-activate="${m.id}" ${active||!Object.keys(m.evaluations).length?'disabled':''}>${active?'✓ Active':'Activate'}</button><a class="btn" href="${endpoint}/models/${m.id}/download">Download .pt</a><button class="btn danger" data-delete-model="${m.id}" ${active?'disabled':''} title="${active?'Active model cannot be deleted':'Delete this model'}">Delete</button></div></article>`;
    }).join('')||emptyState('scan','No custom candidates yet','Train a model, import a compatible .pt file, or download weights from Roboflow.');
    setHTML($('#model-list'),html);
    $$('[data-evaluate]').forEach(button=>button.onclick=async()=>{button.disabled=true;try{await post('/models/'+button.dataset.evaluate+'/evaluate',{dataset:$('#evaluate-dataset').value});message('Evaluation started on held-out validation images.');await loadTraining();}catch(err){message(err.message,true);}finally{button.disabled=false;}});
    $$('[data-activate]').forEach(button=>button.onclick=async()=>{
      const ids=$$('#activation-cameras input:checked').map(i=>i.value);if(!ids.length)return message('Select the cameras that should use this custom model.',true);
      button.disabled=true;
      try{await post('/models/'+button.dataset.activate+'/activate',{cameras:ids});await refreshSettings();message('Custom model activated. Previous model is available for rollback.');await loadTraining();}catch(err){message(err.message,true);}finally{button.disabled=false;}
    });
    $$('[data-delete-model]').forEach(button=>button.onclick=async()=>{
      if(!confirm('Delete this model permanently? This cannot be undone.'))return;
      button.disabled=true;
      try{await api(endpoint+'/models/'+button.dataset.deleteModel,{method:'DELETE'});message('Model deleted.');await loadTraining();}catch(err){message(err.message,true);button.disabled=false;}
    });
  }
  $('#evaluate-dataset').onchange=renderModels;
  async function refreshSettings(){S=await api('/api/settings');renderGrid();renderGlobal();if(!$('#settings').classList.contains('hidden'))renderEditor();await pollStatus();}
  action('#rollback-model',async()=>{await post('/rollback');await refreshSettings();message('Previous custom model and camera selection restored.');});
  const baselineButton=document.createElement('button');baselineButton.className='btn';baselineButton.id='import-current-model';baselineButton.textContent='Add current model for comparison';
  $('#rollback-model').after(baselineButton);
  action('#import-current-model',async()=>{await post('/models/current');message('Current custom model is being copied into the library. Evaluate it on the same dataset as your candidate.');});
  function renderJobs(){
    const finished=training.jobs.some(j=>['completed','failed','cancelled'].includes(j.status));
    const clearBtn=finished?`<div class="row"><span class="spacer"></span><button class="btn" id="clear-jobs">Clear finished jobs</button></div>`:'';
    const html=clearBtn+(training.jobs.slice(0,12).map(j=>`<article class="job-row"><div class="row"><b>${esc(j.kind)}</b><span class="chip ${j.status==='completed'?'reviewed':j.status==='failed'?'alert':''}">${esc(j.status)}</span><span class="spacer"></span>${['queued','running'].includes(j.status)?`<button class="btn" data-cancel-job="${j.id}">Cancel</button>`:''}</div><progress max="100" value="${Math.max(0,Math.min(100,j.progress||0))}" aria-label="${esc(j.kind)} progress"></progress><p class="hint">${esc(j.message)}</p></article>`).join('')||'<p class="hint">Your training, import, upload and download jobs will appear here.</p>');
    setHTML($('#training-jobs'),html);
    if(finished)$('#clear-jobs').onclick=async()=>{try{await post('/jobs/clear');await loadTraining();}catch(err){message(err.message,true);}};
    $$('[data-cancel-job]').forEach(button=>button.onclick=async()=>{try{await post('/jobs/'+button.dataset.cancelJob+'/cancel');await loadTraining();}catch(err){message(err.message,true);}});
  }
  action('#rf-connect',async()=>{const result=await post('/roboflow/connect',{workspace:$('#rf-workspace').value.trim(),project:$('#rf-project').value.trim(),api_key:$('#rf-key').value});$('#rf-key').value='';message('Connected to '+result.project);});
  action('#rf-download-dataset',async()=>{await post('/roboflow/dataset',{version:+$('#rf-version').value});message('Dataset download queued.');});
  action('#rf-download-model',async()=>{await post('/roboflow/model',{model:$('#rf-model').value.trim()});message('Model download queued. Compatibility is checked before registration.');});
  action('#rf-train',async()=>{
    if(!confirm('Start cloud training using your Roboflow account credits for this dataset version?'))return;
    const result=await post('/roboflow/train',{version:+$('#rf-version').value,model_type:$('#rf-model-type').value.trim(),epochs:+$('#rf-epochs').value});message(result.message);
  });
  $('#rf-model-type').placeholder='Optional; blank uses the Roboflow default';
  action('#rf-status',async()=>{
    const result=await api(endpoint+'/roboflow/status?version='+encodeURIComponent($('#rf-version').value));
    $('#rf-cloud-status').textContent=result.trainings.map(t=>`${t.modelType||'Model'}: ${t.status} · ${JSON.stringify(t.modelIds||[])}`).join('\n')||'No training jobs for this version.';
  });
  if(S)window.decorateCameras();
  if(!$('#training').classList.contains('hidden'))loadTraining();
  setInterval(()=>{if(!document.hidden&&!$('#training').classList.contains('hidden'))loadTraining();},2500);
})();
