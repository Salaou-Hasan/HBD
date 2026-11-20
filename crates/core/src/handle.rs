use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// A typed, generational handle.
///
/// Handles are cheap to copy and remain `Copy` regardless of `T`. A handle
/// becomes stale when its slot is freed and reused; checking `is_alive` on the
/// allocator detects that.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    pub const fn from_raw(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    pub const fn index(self) -> u32 {
        self.index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }

    pub const fn to_raw(self) -> (u32, u32) {
        (self.index, self.generation)
    }

    pub const fn dangling() -> Self {
        Self::from_raw(u32::MAX, 0)
    }

    pub const fn is_dangling(self) -> bool {
        self.index == u32::MAX
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

#[derive(Debug)]
struct Slot {
    generation: u32,
    alive: bool,
}

/// Allocates generational handles for a homogeneous slot storage.
///
/// This is the shared handle machinery for assets, physics bodies, and other
/// engine resources. It is intentionally not a container: it only owns the
/// index/generation bookkeeping.
pub struct HandleAllocator<T> {
    slots: Vec<Slot>,
    free: Vec<u32>,
    len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> HandleAllocator<T> {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn allocate(&mut self) -> Handle<T> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(!slot.alive);
            slot.alive = true;
            self.len += 1;
            Handle::from_raw(index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                alive: true,
            });
            self.len += 1;
            Handle::from_raw(index, 0)
        }
    }

    /// Frees a handle. Returns `false` for stale or already-freed handles.
    pub fn free(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slots.get_mut(handle.index() as usize) else {
            return false;
        };

        if !slot.alive || slot.generation != handle.generation() {
            return false;
        }

        slot.alive = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index());
        self.len -= 1;
        true
    }

    pub fn is_alive(&self, handle: Handle<T>) -> bool {
        self.slots
            .get(handle.index() as usize)
            .is_some_and(|slot| slot.alive && slot.generation == handle.generation())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.len = 0;
    }
}

impl<T> Default for HandleAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for HandleAllocator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandleAllocator")
            .field("len", &self.len)
            .field("capacity", &self.slots.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_slots_with_a_new_generation() {
        let mut allocator = HandleAllocator::<u32>::new();

        let first = allocator.allocate();
        assert!(allocator.is_alive(first));
        assert_eq!(allocator.len(), 1);

        assert!(allocator.free(first));
        assert!(!allocator.is_alive(first));
        assert_eq!(allocator.len(), 0);

        let second = allocator.allocate();
        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
        assert!(allocator.is_alive(second));
        assert!(!allocator.is_alive(first));
    }

    #[test]
    fn rejects_stale_handles() {
        let mut allocator = HandleAllocator::<u32>::new();
        let handle = allocator.allocate();
        assert!(allocator.free(handle));
        assert!(!allocator.free(handle));
        assert!(!allocator.is_alive(handle));
    }

    #[test]
    fn dangling_handle_is_distinguishable() {
        let handle = Handle::<u32>::dangling();
        assert!(handle.is_dangling());
        assert!(handle.index() == u32::MAX);
    }
}
