// Service Worker - لغة الإشارة العربية
const CACHE_NAME = 'arsl-v1';

const STATIC_ASSETS = [
  '/',
  '/login',
  '/static/pwa/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  'https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap'
];

// ===== install =====
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// ===== activate =====
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ===== fetch =====
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // API + MediaPipe = network only
  if (
    url.pathname.startsWith('/predict') ||
    url.pathname.startsWith('/collect') ||
    url.pathname.startsWith('/auth') ||
    url.pathname.startsWith('/history') ||
    url.pathname.startsWith('/admin') ||
    url.hostname.includes('mediapipe') ||
    url.hostname.includes('jsdelivr')
  ) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request).then(r => r || caches.match('/')))
  );
});

