//! Camera and projection helpers.
//!
//! The renderer uses right-handed view space and Vulkan's 0..1 clip-space
//! depth range. Naga's coordinate-space adjustment handles the Y flip that
//! Vulkan's framebuffer coordinate system requires.

use hbd_core::math::{Mat4, Vec2, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3) -> Self {
        Self {
            position,
            target,
            ..Default::default()
        }
    }

    /// Returns the combined view-projection matrix for the given aspect ratio.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// View-projection with sub-pixel jitter in NDC units.
    pub fn view_projection_with_jitter(&self, aspect: f32, jitter: Vec2) -> Mat4 {
        let mut projection = self.projection(aspect);
        projection.z_axis.x += jitter.x;
        projection.z_axis.y += jitter.y;
        projection * self.view()
    }

    pub fn view(&self) -> Mat4 {
        look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        perspective_rh(
            self.fov_y_radians,
            aspect.max(0.01),
            self.z_near,
            self.z_far,
        )
    }
}

/// A directional light with an orthographic shadow camera.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    /// Direction from the surface toward the light.
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 1.0,
        }
    }
}

impl DirectionalLight {
    /// Returns an orthographic view-projection matrix covering `extent` units.
    pub fn view_projection(&self, extent: f32) -> Mat4 {
        let eye = self.direction * 24.0;
        let view = look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let projection = orthographic_rh(extent, 0.1, 100.0);
        projection * view
    }
}

/// Right-handed orthographic matrix with Vulkan's 0..1 depth range.
fn orthographic_rh(extent: f32, z_near: f32, z_far: f32) -> Mat4 {
    let m22 = 1.0 / (z_near - z_far);
    let m32 = z_near / (z_near - z_far);

    Mat4::from_cols_array(&[
        1.0 / extent,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / extent,
        0.0,
        0.0,
        0.0,
        0.0,
        m22,
        0.0,
        0.0,
        0.0,
        m32,
        1.0,
    ])
}

/// Right-handed look-at matrix using column-major storage.
fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let forward = (center - eye).normalize();
    let right = forward.cross(up).normalize();
    let camera_up = right.cross(forward);

    Mat4::from_cols_array(&[
        right.x,
        camera_up.x,
        -forward.x,
        0.0,
        right.y,
        camera_up.y,
        -forward.y,
        0.0,
        right.z,
        camera_up.z,
        -forward.z,
        0.0,
        -right.dot(eye),
        -camera_up.dot(eye),
        forward.dot(eye),
        1.0,
    ])
}

/// Right-handed perspective matrix with Vulkan's 0..1 depth range.
fn perspective_rh(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let focal_length = 1.0 / (fov_y_radians * 0.5).tan();
    let m22 = z_far / (z_near - z_far);
    let m32 = (z_near * z_far) / (z_near - z_far);

    Mat4::from_cols_array(&[
        focal_length / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        focal_length,
        0.0,
        0.0,
        0.0,
        0.0,
        m22,
        -1.0,
        0.0,
        0.0,
        m32,
        0.0,
    ])
}
