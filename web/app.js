/**
 * TTM 演示前端（T11）——零构建链：CDN ESM three.js，无 npm / vite / webpack。
 *
 * 关键设计：
 *  1. **rest pose delta 在前端算**：Python 侧只导出「相对源 rest（单位旋转）的局部
 *     四元数」，本文件在加载完 FBX 后读取每根骨的 rest 局部四元数 `q_rest`，按
 *     `q_local(t) = q_rest ∘ q_motion(t)` 合成。关掉「rest 补偿」则直接铺 `q_motion`，
 *     用来目视对比两种方式的差别。
 *  2. **角色归一化**：FBX 可能是 cm（UE）也可能是 m。加载后按包围盒高度缩放到 1.7
 *     个单位，并把脚底对齐 y=0。缩放后 1 个单位 = 1 米，因此根位移直接 `cm / 100`，
 *     与原始单位无关。
 *  3. **兜底可视化**：FBX 不可用时退化成「线框骨架」，页面仍能演示动作与接缝连续性。
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
const CFG = {
  /** 接口基址：与页面同源（FastAPI 同时托管静态文件与 API）。 */
  apiBase: '',
  /** 角色归一化后的身高（单位：米）。 */
  targetHeight: 1.7,
  /** 兜底骨架的骨骼长度（米，未缩放时）。 */
  fallbackBoneLength: 1.0,
  /** 模型朝向修正：BVH 前向为 +Z，导入后按需转动。 */
  modelYaw: Math.PI,
  /** 播放速度（一秒时间线对应多少秒真实时间）。 */
  playbackSpeed: 1.0,
  /** UE 骨名 → 兜底骨架的近似关节位置（用于 FBX 缺失时仍能画出人形）。 */
  fallbackSkeleton: [
    ['root', 0.0, 0.0, 0.0],
    ['pelvis', 0.0, 0.95, 0.0],
    ['spine_01', 0.0, 1.05, 0.0],
    ['spine_02', 0.0, 1.15, 0.0],
    ['spine_03', 0.0, 1.25, 0.0],
    ['neck_01', 0.0, 1.42, 0.0],
    ['head', 0.0, 1.55, 0.0],
    ['clavicle_l', 0.05, 1.40, 0.0],
    ['upperarm_l', 0.18, 1.38, 0.0],
    ['lowerarm_l', 0.42, 1.32, 0.0],
    ['hand_l', 0.63, 1.26, 0.0],
    ['clavicle_r', -0.05, 1.40, 0.0],
    ['upperarm_r', -0.18, 1.38, 0.0],
    ['lowerarm_r', -0.42, 1.32, 0.0],
    ['hand_r', -0.63, 1.26, 0.0],
    ['thigh_l', 0.10, 0.92, 0.0],
    ['calf_l', 0.11, 0.52, 0.0],
    ['foot_l', 0.11, 0.10, 0.0],
    ['ball_l', 0.11, 0.03, 0.08],
    ['thigh_r', -0.10, 0.92, 0.0],
    ['calf_r', -0.11, 0.52, 0.0],
    ['foot_r', -0.11, 0.10, 0.0],
    ['ball_r', -0.11, 0.03, 0.08],
  ],
};

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const state = {
  motion: null,          // ttm-motion/v1 解析结果
  frame: 0,              // 当前帧（浮点，便于插值）
  playing: true,
  restComp: true,
  boneMap: new Map(),    // UE 骨名 → THREE.Bone
  restQuats: new Map(),  // UE 骨名 → THREE.Quaternion（rest 局部旋转）
  motionQuats: new Map(),// UE 骨名 → Float32Array（[F*4]，xyzw 交错）
  rootTrack: null,       // Float32Array [F*3]（cm）
  rootOrigin: new THREE.Vector3(),
  model: null,
  noSkeleton: false,    // 当前角色没有骨骼（如 *_no_bones）：只能静态展示
  wrapper: new THREE.Group(),
  skeletonHelper: null,
  grid: null,
  orbit: true,
  elapsed: 0,
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const dom = {
  statusDot: $('status-dot'),
  statusText: $('status-text'),
  prompt: $('prompt'),
  chips: $('chips'),
  duration: $('duration'),
  durationVal: $('duration-val'),
  guidance: $('guidance'),
  guidanceVal: $('guidance-val'),
  steps: $('steps'),
  stepsVal: $('steps-val'),
  smooth: $('smooth'),
  smoothVal: $('smooth-val'),
  seed: $('seed'),
  generate: $('generate'),
  hint: $('hint'),
  play: $('play'),
  restart: $('restart'),
  character: $('character'),
  scrub: $('scrub'),
  frameLabel: $('frame-label'),
  fpsLabel: $('fps-label'),
  toggleSkeleton: $('toggle-skeleton'),
  toggleRest: $('toggle-rest'),
  toggleGrid: $('toggle-grid'),
  toggleOrbit: $('toggle-orbit'),
  statWindows: $('stat-windows'),
  statSeam: $('stat-seam'),
  statTotal: $('stat-total'),
  statSample: $('stat-sample'),
  statDecode: $('stat-decode'),
  statBlend: $('stat-blend'),
  downloads: $('downloads'),
  canvas: $('canvas'),
  viewport: $('viewport'),
  overlay: $('overlay'),
};

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
const setStatus = (text, kind = '') => {
  dom.statusText.textContent = text;
  dom.statusDot.className = `dot ${kind}`;
};

const setHint = (text, kind = '') => {
  dom.hint.textContent = text;
  dom.hint.className = `hint${kind ? ` ${kind}` : ''}`;
};

const fmt = (value, digits = 1) => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '–';
};

/** 把 FBX/UE 里的骨名归一化：去掉命名空间前缀，转小写。 */
const normalizeBoneName = (raw) => {
  let name = String(raw || '').trim();
  const cut = Math.max(name.lastIndexOf(':'), name.lastIndexOf('|'));
  if (cut >= 0 && cut < name.length - 1) name = name.slice(cut + 1);
  return name.toLowerCase().replace(/^mixamorig/, '').replace(/[_\s]?(bn|bone)$/, '');
};

/** 四元数数组（xyzw）→ THREE.Quaternion。 */
const quatAt = (array, offset, target) => target.set(
  array[offset], array[offset + 1], array[offset + 2], array[offset + 3],
);

// ---------------------------------------------------------------------------
// three.js 场景
// ---------------------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a0e);
scene.fog = new THREE.Fog(0x070a0e, 9, 26);

const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 120);
camera.position.set(2.6, 1.9, 3.6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.95, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.2;
controls.maxDistance = 16;
controls.maxPolarAngle = Math.PI * 0.495;

// 灯光
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x24303f, 1.15));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(3.2, 6.0, 4.0);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 22;
keyLight.shadow.camera.left = -5;
keyLight.shadow.camera.right = 5;
keyLight.shadow.camera.top = 6;
keyLight.shadow.camera.bottom = -1;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x4dd4ac, 0.75);
rimLight.position.set(-4.0, 2.4, -3.6);
scene.add(rimLight);

// 地面 + 网格
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(14, 64).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: 0x0e131a, roughness: 0.95, metalness: 0.0 }),
);
ground.receiveShadow = true;
scene.add(ground);

state.grid = new THREE.GridHelper(24, 48, 0x2b3a4d, 0x1a2430);
state.grid.position.y = 0.001;
scene.add(state.grid);

state.wrapper.position.set(0, 0, 0);
scene.add(state.wrapper);

/** 视口尺寸自适应。 */
const resize = () => {
  const width = dom.viewport.clientWidth;
  const height = dom.viewport.clientHeight;
  if (width <= 0 || height <= 0) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
};

window.addEventListener('resize', resize);

// ---------------------------------------------------------------------------
// 角色加载：FBX → 归一化 → 抽取 rest pose
// ---------------------------------------------------------------------------
/** 清空 wrapper 里旧的模型与辅助对象。 */
const clearModel = () => {
  while (state.wrapper.children.length > 0) {
    const child = state.wrapper.children.pop();
    child.traverse?.((node) => {
      if (node.isMesh) {
        node.geometry?.dispose?.();
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        materials.forEach((material) => material?.dispose?.());
      }
    });
  }
  state.model = null;
  state.boneMap = new Map();
  state.restQuats = new Map();
  if (state.skeletonHelper) {
    scene.remove(state.skeletonHelper);
    state.skeletonHelper.dispose?.();
    state.skeletonHelper = null;
  }
};

/** 递归收集骨骼，按归一化骨名建索引。 */
const collectBones = (object3d) => {
  const bones = new Map();
  object3d.traverse((node) => {
    if (node.isBone) bones.set(normalizeBoneName(node.name), node);
  });
  return bones;
};

/** 抽取 rest 局部四元数（模型加载完成那一刻的局部姿态就是 rest pose）。 */
const captureRestPose = (bones) => {
  const rest = new Map();
  bones.forEach((bone, name) => rest.set(name, bone.quaternion.clone()));
  return rest;
};

/** characters.json / /characters 下发的清单：FBX 名 → { has_bones, bone_names }。 */
const manifestByName = new Map();

/** 从 /characters 的 url 里取文件名（清单按名字对账）。 */
const fileNameOf = (url) => decodeURIComponent(String(url || '').split('/').pop() || '');

/**
 * 按 manifest 校验已加载骨架的命名覆盖率（F-18）。
 * 返回空串表示通过；否则返回可读的警告片段。
 */
const checkBoneCoverage = (declared, bones) => {
  if (!declared || !declared.has_bones) return '';
  const expected = Array.isArray(declared.bone_names) ? declared.bone_names : [];
  if (expected.length === 0) return '';
  const covered = expected.filter((name) => bones.has(normalizeBoneName(name))).length;
  const ratio = covered / expected.length;
  return ratio < 0.6
    ? ` · 骨骼命名覆盖 ${covered}/${expected.length}（<60%，部分动作可能驱动不了）`
    : '';
};

/**
 * 归一化模型：缩放到目标身高、脚底对齐 y=0、朝向修正。
 * 缩放后 1 个场景单位 = 1 米（与原始单位无关），因此根位移直接用 cm/100。
 */
const normalizeModel = (object3d) => {
  const box = new THREE.Box3().setFromObject(object3d);
  const size = new THREE.Vector3();
  box.getSize(size);
  const height = Math.max(size.y, 1e-6);
  const scale = CFG.targetHeight / height;
  object3d.scale.multiplyScalar(scale);
  object3d.updateMatrixWorld(true);

  const scaledBox = new THREE.Box3().setFromObject(object3d);
  const center = new THREE.Vector3();
  scaledBox.getCenter(center);
  // 水平居中 + 脚底贴地
  object3d.position.x -= center.x;
  object3d.position.z -= center.z;
  object3d.position.y -= scaledBox.min.y;
  object3d.rotation.y += CFG.modelYaw;
  object3d.updateMatrixWorld(true);
  return { scale, nativeHeight: height };
};

/** FBX 加载失败时的兜底：用 UE 骨名 + 近似关节位置画一个线框人。 */
const buildFallbackRig = () => {
  const bones = new Map();
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: 0x4dd4ac, roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.85,
  });

  const index = new Map();
  CFG.fallbackSkeleton.forEach(([name, x, y, z], position) => {
    const bone = new THREE.Bone();
    bone.name = name;
    bone.position.set(x, y, z);
    index.set(name, { bone, position });
    bones.set(name, bone);
  });

  // 父级关系按 UE 骨架（与 constants.UE_PARENTS 一致）
  const parentOf = {
    pelvis: 'root', spine_01: 'pelvis', spine_02: 'spine_01', spine_03: 'spine_02',
    neck_01: 'spine_03', head: 'neck_01',
    clavicle_l: 'spine_03', upperarm_l: 'clavicle_l', lowerarm_l: 'upperarm_l', hand_l: 'lowerarm_l',
    clavicle_r: 'spine_03', upperarm_r: 'clavicle_r', lowerarm_r: 'upperarm_r', hand_r: 'lowerarm_r',
    thigh_l: 'pelvis', calf_l: 'thigh_l', foot_l: 'calf_l', ball_l: 'foot_l',
    thigh_r: 'pelvis', calf_r: 'thigh_r', foot_r: 'calf_r', ball_r: 'foot_r',
  };

  // 先把骨骼挂成树：子骨位置转成相对父骨的偏移
  Object.entries(parentOf).forEach(([child, parent]) => {
    const childBone = index.get(child).bone;
    const parentBone = index.get(parent).bone;
    const [cx, cy, cz] = [childBone.position.x, childBone.position.y, childBone.position.z];
    const [px, py, pz] = [parentBone.position.x, parentBone.position.y, parentBone.position.z];
    childBone.position.set(cx - px, cy - py, cz - pz);
    parentBone.add(childBone);
  });

  const rootBone = index.get('root').bone;
  rootBone.position.set(0, 0, 0);
  group.add(rootBone);

  // 用线段把骨肉画出来（父骨 → 子骨）
  const positions = [];
  Object.entries(parentOf).forEach(([child, parent]) => {
    const a = index.get(parent);
    const b = index.get(child);
    positions.push(a.bone.position.x, a.bone.position.y, a.bone.position.z);
    positions.push(b.bone.position.x, b.bone.position.y, b.bone.position.z);
  });
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0x4dd4ac, transparent: true, opacity: 0.9 }),
  );
  group.add(lines);

  // 头部与骨盆给个体积感
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 16), material);
  head.position.copy(index.get('head').bone.position).add(
    new THREE.Vector3(0, index.get('neck_01').bone.position.y, 0),
  );
  head.castShadow = true;
  group.add(head);

  return { group, bones };
};

/** 加载指定角色；返回 { bones, label }。 */
const loadCharacter = async (url) => {
  const loader = new FBXLoader();
  const object3d = await loader.loadAsync(url);
  object3d.traverse((node) => {
    if (node.isMesh || node.isSkinnedMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
      node.frustumCulled = false; // 蒙皮包围盒不准，避免动作幅度大时被裁掉
    }
  });
  const info = normalizeModel(object3d);
  state.wrapper.add(object3d);
  state.model = object3d;
  const bones = collectBones(object3d);
  return { bones, label: `FBX（原生高度 ${info.nativeHeight.toFixed(1)}）` };
};

/** 初始化角色：优先 FBX，解析失败退兜底线框，无骨骼则静态展示并明确告知。 */
const setupCharacter = async (preferred, displayName = '') => {
  clearModel();
  state.noSkeleton = false;
  let bones;
  let label;
  let coverageNote = '';
  try {
    if (!preferred) throw new Error('没有可用角色');
    const result = await loadCharacter(preferred);
    bones = result.bones;
    label = result.label;
    // 无骨骼角色（如 SK_Sun_CASUAL_skirt_no_bones）：能加载出静态网格但一个 Bone
    // 都没有。不能按「加载失败」退回线框（那样反而把模型藏了），也不能白屏——
    // 正确做法是照常展示静态模型，同时明确告知「这个角色驱动不了动作」。
    if (bones.size === 0) {
      state.noSkeleton = true;
      label = `${displayName || '所选 FBX'} · 无骨骼，仅静态展示`;
    } else {
      // manifest 校验（F-18）：声明有骨骼的，检查骨架命名覆盖率
      const declared = manifestByName.get(fileNameOf(preferred));
      coverageNote = checkBoneCoverage(declared, bones);
    }
  } catch (error) {
    console.warn('[ttm] FBX 加载失败，退化为线框骨架：', error);
    const fallback = buildFallbackRig();
    state.wrapper.add(fallback.group);
    state.model = fallback.group;
    bones = fallback.bones;
    label = '兜底线框骨架（FBX 不可用）';
  }

  state.boneMap = bones;
  state.restQuats = captureRestPose(bones);

  // 骨骼辅助线（只在有真实骨骼时才有意义）
  if (state.model && bones.size > 4) {
    state.skeletonHelper = new THREE.SkeletonHelper(state.model);
    state.skeletonHelper.material.linewidth = 1;
    state.skeletonHelper.visible = dom.toggleSkeleton.checked;
    scene.add(state.skeletonHelper);
  } else {
    state.skeletonHelper = null;
  }

  const matched = state.motion && !state.noSkeleton ? countMatchedBones() : 0;
  if (state.noSkeleton) {
    setStatus(`就绪 · ${label}`, 'warn');
    setHint(
      `「${displayName || '该角色'}」没有骨骼数据，无法驱动动作，仅展示静态模型；`
      + '要播放生成结果请切换到其它角色。',
      'warn',
    );
  } else if (coverageNote) {
    setStatus(`就绪 · ${label}${coverageNote}`, 'warn');
  } else {
    setStatus(`就绪 · ${label}${state.motion ? ` · 匹配 ${matched} 骨` : ''}`, 'ok');
  }
  dom.overlay.classList.add('hidden');
  return label;
};

/** 统计 motion 里有多少骨名能在角色骨架上找到。 */
const countMatchedBones = () => {
  if (!state.motion) return 0;
  let matched = 0;
  state.motion.tracks.forEach((track) => {
    if (state.boneMap.has(normalizeBoneName(track.bone))) matched += 1;
  });
  return matched;
};

// ---------------------------------------------------------------------------
// 接口调用
// ---------------------------------------------------------------------------
const requestJson = async (path, options) => {
  const response = await fetch(`${CFG.apiBase}${path}`, options);
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch (error) {
    body = { detail: text };
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body;
};

/** 探活 + 拉取示例指令与角色列表。 */
const bootstrap = async () => {
  setStatus('检查服务…');
  try {
    const health = await requestJson('/health');
    const weights = health.weights || {};
    const missing = Object.entries(weights)
      .filter(([, value]) => !value.exists)
      .map(([key]) => key);
    if (health.generator && health.generator.loaded) {
      setStatus(`服务就绪 · ${health.device} · 模型已加载`, 'ok');
    } else if (missing.length > 0) {
      setStatus(`服务在线 · 缺少权重：${missing.join(', ')}`, 'bad');
      setHint(`权重缺失：${missing.join(', ')}。先跑 03/04 训练脚本，再点「生成动作」。`, 'error');
    } else {
      setStatus(`服务在线 · ${health.device} · 模型待加载`, 'ok');
    }
  } catch (error) {
    setStatus('服务不可用（用 file:// 打开？请走 FastAPI 的 8000/8765 端口）', 'bad');
    setHint(`无法连接后端：${error.message}`, 'error');
    return;
  }

  try {
    const prompts = await requestJson('/prompts');
    renderChips(prompts.prompts || []);
    dom.duration.max = String(prompts.max_duration_s || 12);
  } catch (error) {
    console.warn('[ttm] /prompts 失败', error);
  }

  let preferred = '';
  try {
    const characters = await requestJson('/characters');
    const defaultName = characters.default || '';
    dom.character.innerHTML = '';
    (characters.items || []).forEach((item) => {
      const option = document.createElement('option');
      option.value = item.url;
      option.dataset.name = item.name;
      // manifest 信息存起来，setupCharacter 加载完做命名覆盖率校验（F-18）
      manifestByName.set(item.name, item);
      // 无骨骼角色在列表里就标出来，别等用户选完加载完才发现驱动不了
      const noBones = item.has_bones === false || /no[_-]?bones/i.test(item.name);
      option.textContent = noBones ? `${item.name}（无骨骼 · 仅静态展示）` : item.name;
      if (item.name === defaultName) option.selected = true;
      dom.character.appendChild(option);
    });

    // ⚠ 这里**故意**保留裸文件名，不要改成 item.url。
    // `/characters` 下发的 default 是裸文件名，原来当 URL 用会请求 /SK_xxx.FBX 拿到 404，
    // 于是走下面的 catch 退化成线框骨架（能看，只是没有网格）。
    // 一旦改成正确的 /characters/<name>，请求会真的成功，然后 FBXLoader 在解析这批
    // 资产的 morph target 时把渲染进程打崩（RangeError: Array buffer allocation failed
    // → Page crashed，JS 侧 catch 不到，整页白屏）。四个 FBX 全部复现，换 three 0.175/0.180 也一样。
    // 正确修法是先把资产转成 glTF 再用 GLTFLoader（见 README 已知限制 8），不是改这一行。
    preferred = characters.default || '';
    if (!characters.items || characters.items.length === 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '（未找到 FBX）';
      dom.character.appendChild(option);
    }
  } catch (error) {
    console.warn('[ttm] /characters 失败', error);
  }

  await setupCharacter(preferred);
};

// ---------------------------------------------------------------------------
// 动作数据装配
// ---------------------------------------------------------------------------
/**
 * 解析 ``ttm-motion/v1``：把每条 track 的旋转拷成连续 Float32Array，
 * 并按归一化骨名建索引（避免每帧查对象）。
 */
const ingestMotion = (payload) => {
  const tracks = Array.isArray(payload.tracks) ? payload.tracks : [];
  const motionQuats = new Map();
  tracks.forEach((track) => {
    const flat = new Float32Array(track.rotations_xyzw.length * 4);
    track.rotations_xyzw.forEach((quaternion, frame) => {
      flat[frame * 4 + 0] = quaternion[0];
      flat[frame * 4 + 1] = quaternion[1];
      flat[frame * 4 + 2] = quaternion[2];
      flat[frame * 4 + 3] = quaternion[3];
    });
    motionQuats.set(normalizeBoneName(track.bone), flat);
  });

  const positions = Array.isArray(payload.root_positions_cm) ? payload.root_positions_cm : [];
  // ground_origin（M4 新增）：ttm-motion/v1 的 root_positions_cm 保留**原始世界坐标**
  // （楼梯类动作根高会随楼层累加到 4 米+），offset 写在 meta 里。web 预览默认
  // 地面锚定——在这里把 Y 减掉 offset，首帧足触地点落在 y=0。
  const groundMeta = payload.ground_origin && typeof payload.ground_origin === 'object'
    ? payload.ground_origin
    : {};
  const groundOffset = groundMeta.applied ? (Number(groundMeta.offset_cm) || 0) : 0;
  const rootTrack = new Float32Array(positions.length * 3);
  positions.forEach((position, frame) => {
    rootTrack[frame * 3 + 0] = position[0];
    rootTrack[frame * 3 + 1] = position[1] - groundOffset;
    rootTrack[frame * 3 + 2] = position[2];
  });

  state.motion = { ...payload, tracks, nFrames: Math.max(0, payload.n_frames | 0) };
  state.motionQuats = motionQuats;
  state.rootTrack = rootTrack;
  state.rootOrigin.set(
    rootTrack.length >= 3 ? rootTrack[0] : 0,
    rootTrack.length >= 3 ? rootTrack[1] : 0,
    rootTrack.length >= 3 ? rootTrack[2] : 0,
  );
  state.frame = 0;
  state.elapsed = 0;
  dom.scrub.min = '0';
  dom.scrub.max = String(Math.max(0, state.motion.nFrames - 1));
  dom.scrub.value = '0';
  dom.fpsLabel.textContent = `${payload.fps || 30} fps`;
};

/**
 * 把第 ``frame`` 帧的姿态铺到骨骼上。
 *
 * ``restComp = true``：``q_local = q_rest ∘ q_motion``（rest pose delta 补偿）；
 * ``restComp = false``：``q_local = q_motion``（直接铺，A-pose 角色会明显歪掉）。
 */
const applyFrame = (frameIndex) => {
  if (!state.motion) return;
  const total = state.motion.nFrames;
  if (total <= 0) return;
  const frame = Math.min(Math.max(frameIndex, 0), total - 1);
  const base = frame * 4;
  const target = new THREE.Quaternion();

  state.motionQuats.forEach((flat, name) => {
    const bone = state.boneMap.get(name);
    if (!bone || flat.length < base + 4) return;
    quatAt(flat, base, target);
    if (state.restComp) {
      const rest = state.restQuats.get(name);
      if (rest) bone.quaternion.copy(rest).multiply(target);
      else bone.quaternion.copy(target);
    } else {
      bone.quaternion.copy(target);
    }
  });

  // 根位移：cm → m（归一化后 1 单位 = 1 米），并取首帧为原点。
  // 无骨骼角色不铺根位移——模型动不了，整体平移只会看起来像「漂移」。
  if (!state.noSkeleton && state.rootTrack && state.rootTrack.length >= 3) {
    const index = frame * 3;
    state.wrapper.position.set(
      (state.rootTrack[index + 0] - state.rootOrigin.x) / 100,
      (state.rootTrack[index + 1] - state.rootOrigin.y) / 100,
      (state.rootTrack[index + 2] - state.rootOrigin.z) / 100,
    );
  }

  dom.frameLabel.textContent = `帧 ${frame + 1} / ${total}`;
  dom.scrub.value = String(frame);
};

// ---------------------------------------------------------------------------
// 生成
// ---------------------------------------------------------------------------
const readNumber = (element, fallback) => {
  const value = Number(element.value);
  return Number.isFinite(value) ? value : fallback;
};

const generate = async () => {
  const prompt = dom.prompt.value.trim();
  if (!prompt) {
    setHint('请先输入中文指令。', 'error');
    return;
  }
  dom.generate.disabled = true;
  dom.generate.textContent = '生成中…';
  dom.overlay.textContent = '正在生成动作（首次需加载模型）…';
  dom.overlay.classList.remove('hidden');
  setHint('首次生成会加载模型，约 10–20 秒；之后秒级响应。');

  const body = {
    prompt,
    duration_s: readNumber(dom.duration, 5),
    steps: Math.round(readNumber(dom.steps, 50)),
    guidance: readNumber(dom.guidance, 7.5),
    smooth: readNumber(dom.smooth, 0.2),
    seed: Math.round(readNumber(dom.seed, 42)),
    export: 'json',
  };

  try {
    const result = await requestJson('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    ingestMotion(result.motion || {});
    renderStats(result);
    renderDownloads(result);
    state.playing = true;
    dom.play.textContent = '❚❚';
    applyFrame(0);
    setStatus(`已生成 · ${result.n_frames} 帧 / ${result.n_windows} 窗`, 'ok');
    setHint(`生成完成：${result.n_frames} 帧，用时 ${fmt(result.timing.total_ms, 0)} ms。`);
  } catch (error) {
    setHint(`生成失败：${error.message}`, 'error');
    setStatus('生成失败', 'bad');
  } finally {
    dom.generate.disabled = false;
    dom.generate.textContent = '生成动作';
    dom.overlay.classList.add('hidden');
  }
};

const renderStats = (result) => {
  const timing = result.timing || {};
  const jumps = Array.isArray(result.seam_jumps) ? result.seam_jumps : [];
  dom.statWindows.textContent = String(result.n_windows ?? '–');
  dom.statSeam.textContent = jumps.length > 0
    ? `${fmt(Math.max(...jumps), 1)} cm/s`
    : '单窗';
  dom.statTotal.textContent = `${fmt(timing.total_ms, 0)} ms`;
  dom.statSample.textContent = `${fmt(timing.sample_ms, 0)} ms`;
  dom.statDecode.textContent = `${fmt(timing.decode_ms, 0)} ms`;
  dom.statBlend.textContent = `${fmt(timing.blend_ms, 0)} ms`;
};

const renderDownloads = (result) => {
  dom.downloads.innerHTML = '';
  const entries = [
    ['Motion JSON', result.json_url],
    ['BVH', result.bvh_url],
  ].filter(([, url]) => Boolean(url));
  entries.forEach(([label, url]) => {
    const link = document.createElement('a');
    link.href = url;
    link.textContent = `下载 ${label}`;
    link.setAttribute('download', '');
    dom.downloads.appendChild(link);
  });
};

const renderChips = (prompts) => {
  dom.chips.innerHTML = '';
  prompts.slice(0, 6).forEach((prompt) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = prompt;
    chip.addEventListener('click', () => {
      dom.prompt.value = prompt;
    });
    dom.chips.appendChild(chip);
  });
};

// ---------------------------------------------------------------------------
// UI 绑定
// ---------------------------------------------------------------------------
const bindRange = (input, output, digits) => {
  const sync = () => {
    output.textContent = digits === 0
      ? String(Math.round(Number(input.value)))
      : Number(input.value).toFixed(digits);
  };
  input.addEventListener('input', sync);
  sync();
};

bindRange(dom.duration, dom.durationVal, 1);
bindRange(dom.guidance, dom.guidanceVal, 1);
bindRange(dom.steps, dom.stepsVal, 0);
bindRange(dom.smooth, dom.smoothVal, 2);

dom.generate.addEventListener('click', generate);

dom.prompt.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') generate();
});

dom.play.addEventListener('click', () => {
  state.playing = !state.playing;
  dom.play.textContent = state.playing ? '❚❚' : '▶';
});

dom.restart.addEventListener('click', () => {
  state.frame = 0;
  state.elapsed = 0;
  applyFrame(0);
});

dom.scrub.addEventListener('input', () => {
  state.playing = false;
  dom.play.textContent = '▶';
  state.frame = Number(dom.scrub.value);
  applyFrame(Math.round(state.frame));
});

dom.toggleSkeleton.addEventListener('change', () => {
  if (state.skeletonHelper) state.skeletonHelper.visible = dom.toggleSkeleton.checked;
});

dom.toggleRest.addEventListener('change', () => {
  state.restComp = dom.toggleRest.checked;
  applyFrame(Math.round(state.frame));
});

dom.toggleGrid.addEventListener('change', () => {
  state.grid.visible = dom.toggleGrid.checked;
});

dom.toggleOrbit.addEventListener('change', () => {
  state.orbit = dom.toggleOrbit.checked;
  controls.autoRotate = state.orbit;
  controls.autoRotateSpeed = 0.55;
});

dom.character.addEventListener('change', async () => {
  const url = dom.character.value;
  if (!url) return;
  const name = dom.character.selectedOptions[0]?.dataset.name || '';
  dom.overlay.textContent = '正在切换角色…';
  dom.overlay.classList.remove('hidden');
  await setupCharacter(url, name);
  if (state.motion && !state.noSkeleton) applyFrame(Math.round(state.frame));
});

// ---------------------------------------------------------------------------
// 渲染循环
// ---------------------------------------------------------------------------
let lastTime = performance.now();

const animate = () => {
  const now = performance.now();
  const deltaSeconds = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  if (state.motion && state.playing) {
    const fps = state.motion.fps || 30;
    state.elapsed += deltaSeconds * CFG.playbackSpeed;
    const total = state.motion.nFrames;
    const frame = Math.floor(state.elapsed * fps);
    if (total > 0) {
      if (frame >= total) {
        state.elapsed = 0; // 循环播放
        state.frame = 0;
      } else {
        state.frame = frame;
      }
      applyFrame(Math.round(state.frame));
    }
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
};

resize();
controls.autoRotate = state.orbit;
controls.autoRotateSpeed = 0.55;
animate();
bootstrap();

// 便于在控制台调试
window.TTM = { state, CFG, scene, camera, applyFrame, ingestMotion };
