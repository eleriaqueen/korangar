use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use cgmath::Vector2;
#[cfg(feature = "debug")]
use korangar_debug::logging::{print_debug, Colorize};
#[cfg(feature = "debug")]
use korangar_debug::profile_block;
use wgpu::util::StagingBelt;
use wgpu::{
    Adapter, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Extent3d, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout,
    Instance, Maintain, Origin3d, Queue, SurfaceTexture, TextureAspect, TextureFormat, TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::{
    Capabilities, FramePacer, FrameStage, GlobalContext, LimitFramerate, Prepare, PresentModeInfo, ShadowDetail, Surface,
    TextureSamplerType,
};
use crate::graphics::instruction::RenderInstruction;
use crate::graphics::passes::*;
use crate::interface::layout::ScreenSize;
use crate::loaders::TextureLoader;
use crate::NUMBER_OF_POINT_LIGHTS_WITH_SHADOWS;

pub struct GraphicsEngineDescriptor {
    pub capabilities: Capabilities,
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub texture_loader: Arc<TextureLoader>,
    pub picker_value: Arc<AtomicU64>,
}

/// Bind Group layout:
///
/// The safe default limit for bound bind-groups is 4.
/// When a set of a specific level is not bound, the bindings below it will
/// move up a level.
///
/// Set 0: Global Bindings
///  0 => Uniforms Buffer
///  1 => Nearest Sampler
///  2 => Linear Sampler
///  3 => Texture Sampler
///
/// Set 1: Pass Bindings (For example point shadow project view matrices)
///
/// Set 2: Dispatcher / Drawer Bindings (For example indirection buffers)
///
/// Set 3: Resource Bindings (for example a texture group of a map)
///
/// Push Constants: Draw Data (up to 128 KiB)
pub struct GraphicsEngine {
    capabilities: Capabilities,
    instance: Instance,
    adapter: Adapter,
    device: Arc<Device>,
    queue: Arc<Queue>,
    staging_belt: StagingBelt,
    surface: Option<Surface>,
    frame_pacer: FramePacer,
    cpu_stage: FrameStage<Instant>,
    limit_framerate: bool,
    previous_surface_texture_format: Option<TextureFormat>,
    texture_loader: Arc<TextureLoader>,
    engine_context: Option<EngineContext>,
    picker_value: Arc<AtomicU64>,
}

struct EngineContext {
    global_context: GlobalContext,

    interface_render_pass_context: InterfaceRenderPassContext,
    picker_render_pass_context: PickerRenderPassContext,
    directional_shadow_pass_context: DirectionalShadowRenderPassContext,
    point_shadow_pass_context: PointShadowRenderPassContext,
    geometry_pass_context: GeometryRenderPassContext,
    screen_pass_context: ScreenRenderPassContext,

    interface_rectangle_drawer: InterfaceRectangleDrawer,
    picker_entity_drawer: PickerEntityDrawer,
    picker_tile_drawer: PickerTileDrawer,
    directional_shadow_model_drawer: DirectionalShadowModelDrawer,
    directional_shadow_entity_drawer: DirectionalShadowEntityDrawer,
    directional_shadow_indicator_drawer: DirectionalShadowIndicatorDrawer,
    point_shadow_entity_drawer: PointShadowEntityDrawer,
    point_shadow_model_drawer: PointShadowModelDrawer,
    point_shadow_indicator_drawer: PointShadowIndicatorDrawer,
    geometry_model_drawer: GeometryModelDrawer,
    geometry_indicator_drawer: GeometryIndicatorDrawer,
    geometry_entity_drawer: GeometryEntityDrawer,
    geometry_water_drawer: GeometryWaterDrawer,
    screen_ambient_light_drawer: ScreenAmbientLightDrawer,
    screen_directional_light_drawer: ScreenDirectionalLightDrawer,
    screen_point_light_drawer: ScreenPointLightDrawer,
    screen_water_light_drawer: ScreenWaterLightDrawer,
    screen_rectangle_drawer: ScreenRectangleDrawer,
    screen_effect_drawer: ScreenEffectDrawer,
    screen_overlay_drawer: ScreenOverlayDrawer,

    #[cfg(feature = "debug")]
    picker_marker_drawer: PickerMarkerDrawer,
    #[cfg(feature = "debug")]
    screen_aabb_drawer: ScreenAabbDrawer,
    #[cfg(feature = "debug")]
    screen_circle_drawer: ScreenCircleDrawer,
    #[cfg(feature = "debug")]
    screen_buffer_drawer: ScreenBufferDrawer,
}

impl GraphicsEngine {
    pub fn initialize(descriptor: GraphicsEngineDescriptor) -> GraphicsEngine {
        let staging_belt = StagingBelt::new(1048576); // 1 MiB
        let mut frame_pacer = FramePacer::new(60.0);
        let cpu_stage = frame_pacer.create_frame_stage(Instant::now());

        Self {
            capabilities: descriptor.capabilities,
            instance: descriptor.instance,
            adapter: descriptor.adapter,
            device: descriptor.device,
            queue: descriptor.queue,
            staging_belt,
            surface: None,
            frame_pacer,
            cpu_stage,
            limit_framerate: false,
            previous_surface_texture_format: None,
            texture_loader: descriptor.texture_loader,
            engine_context: None,
            picker_value: descriptor.picker_value,
        }
    }

    pub fn on_resume(
        &mut self,
        window: Arc<Window>,
        triple_buffering: bool,
        vsync: bool,
        limit_framerate: LimitFramerate,
        shadow_detail: ShadowDetail,
        texture_sampler_type: TextureSamplerType,
    ) {
        self.set_limit_framerate(limit_framerate);

        // Android devices need to drop the surface on suspend, so we might need to
        // re-create it.
        if self.surface.is_none() {
            time_phase!("create surface", {
                let screen_size: ScreenSize = window.inner_size().max(PhysicalSize::new(1, 1)).into();
                let raw_surface = self.instance.create_surface(window).unwrap();
                let surface = Surface::new(
                    &self.adapter,
                    self.device.clone(),
                    raw_surface,
                    screen_size.width as u32,
                    screen_size.height as u32,
                    triple_buffering,
                    vsync,
                );
                let surface_texture_format = surface.format();

                if self.previous_surface_texture_format != Some(surface_texture_format) {
                    self.previous_surface_texture_format = Some(surface_texture_format);
                    self.engine_context = None;

                    time_phase!("create contexts", {
                        let global_context = GlobalContext::new(
                            &self.device,
                            &self.queue,
                            &self.texture_loader,
                            surface_texture_format,
                            screen_size,
                            shadow_detail,
                            texture_sampler_type,
                        );

                        let interface_render_pass_context =
                            InterfaceRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                        let picker_render_pass_context =
                            PickerRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                        let directional_shadow_pass_context =
                            DirectionalShadowRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                        let point_shadow_pass_context =
                            PointShadowRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                        let geometry_pass_context =
                            GeometryRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                        let screen_pass_context =
                            ScreenRenderPassContext::new(&self.device, &self.queue, &self.texture_loader, &global_context);
                    });

                    time_phase!("create computer and drawer", {
                        let interface_rectangle_drawer = InterfaceRectangleDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &interface_render_pass_context,
                        );
                        let picker_entity_drawer = PickerEntityDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &picker_render_pass_context,
                        );
                        let picker_tile_drawer = PickerTileDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &picker_render_pass_context,
                        );
                        let directional_shadow_model_drawer = DirectionalShadowModelDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &directional_shadow_pass_context,
                        );
                        let directional_shadow_entity_drawer = DirectionalShadowEntityDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &directional_shadow_pass_context,
                        );
                        let directional_shadow_indicator_drawer = DirectionalShadowIndicatorDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &directional_shadow_pass_context,
                        );
                        let point_shadow_model_drawer = PointShadowModelDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &point_shadow_pass_context,
                        );
                        let point_shadow_entity_drawer = PointShadowEntityDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &point_shadow_pass_context,
                        );
                        let point_shadow_indicator_drawer = PointShadowIndicatorDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &point_shadow_pass_context,
                        );
                        let geometry_model_drawer = GeometryModelDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &geometry_pass_context,
                        );
                        let geometry_entity_drawer = GeometryEntityDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &geometry_pass_context,
                        );
                        let geometry_indicator_drawer = GeometryIndicatorDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &geometry_pass_context,
                        );
                        let geometry_water_drawer = GeometryWaterDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &geometry_pass_context,
                        );
                        let screen_ambient_light_drawer = ScreenAmbientLightDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_directional_light_drawer = ScreenDirectionalLightDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_point_light_drawer = ScreenPointLightDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_water_light_drawer = ScreenWaterLightDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_rectangle_drawer = ScreenRectangleDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_effect_drawer = ScreenEffectDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        let screen_overlay_drawer = ScreenOverlayDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );

                        #[cfg(feature = "debug")]
                        let picker_marker_drawer = PickerMarkerDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &picker_render_pass_context,
                        );
                        #[cfg(feature = "debug")]
                        let screen_aabb_drawer = ScreenAabbDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        #[cfg(feature = "debug")]
                        let screen_circle_drawer = ScreenCircleDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                        #[cfg(feature = "debug")]
                        let screen_buffer_drawer = ScreenBufferDrawer::new(
                            &self.capabilities,
                            &self.device,
                            &self.queue,
                            &global_context,
                            &screen_pass_context,
                        );
                    });

                    self.engine_context = Some(EngineContext {
                        global_context,
                        interface_render_pass_context,
                        picker_render_pass_context,
                        directional_shadow_pass_context,
                        point_shadow_pass_context,
                        geometry_pass_context,
                        screen_pass_context,
                        interface_rectangle_drawer,
                        picker_entity_drawer,
                        picker_tile_drawer,
                        directional_shadow_model_drawer,
                        directional_shadow_entity_drawer,
                        directional_shadow_indicator_drawer,
                        point_shadow_model_drawer,
                        point_shadow_indicator_drawer,
                        point_shadow_entity_drawer,
                        geometry_model_drawer,
                        geometry_entity_drawer,
                        geometry_water_drawer,
                        geometry_indicator_drawer,
                        screen_ambient_light_drawer,
                        screen_directional_light_drawer,
                        screen_point_light_drawer,
                        screen_water_light_drawer,
                        screen_rectangle_drawer,
                        screen_effect_drawer,
                        screen_overlay_drawer,
                        #[cfg(feature = "debug")]
                        picker_marker_drawer,
                        #[cfg(feature = "debug")]
                        screen_aabb_drawer,
                        #[cfg(feature = "debug")]
                        screen_circle_drawer,
                        #[cfg(feature = "debug")]
                        screen_buffer_drawer,
                    })
                }

                self.surface = Some(surface);

                #[cfg(feature = "debug")]
                print_debug!("created {}", "surface".magenta());
            });
        }
    }

    pub fn on_suspended(&mut self) {
        // Android devices are expected to drop their surface view.
        if cfg!(target_os = "android") {
            self.surface = None;
        }
    }

    pub fn on_resize(&mut self, screen_size: ScreenSize) {
        if let Some(surface) = self.surface.as_mut() {
            surface.update_window_size(screen_size);
        }
    }

    pub fn set_vsync(&mut self, enabled: bool) {
        if let Some(surface) = self.surface.as_mut() {
            surface.set_vsync(enabled);
        }
    }

    pub fn set_limit_framerate(&mut self, limit_framerate: LimitFramerate) {
        match limit_framerate {
            LimitFramerate::Unlimited => {
                self.limit_framerate = false;
            }
            LimitFramerate::Limit(rate) => {
                self.limit_framerate = true;
                self.frame_pacer.set_monitor_frequency(f64::from(rate));
            }
        }
    }

    pub fn set_triple_buffering(&mut self, enabled: bool) {
        if let Some(surface) = self.surface.as_mut() {
            surface.set_triple_buffering(enabled);
        }
    }

    pub fn set_texture_sampler_type(&mut self, texture_sampler_type: TextureSamplerType) {
        if let Some(engine_context) = self.engine_context.as_mut() {
            engine_context
                .global_context
                .update_texture_sampler(&self.device, texture_sampler_type);
        }
    }

    pub fn set_shadow_detail(&mut self, shadow_detail: ShadowDetail) {
        if let Some(engine_context) = self.engine_context.as_mut() {
            engine_context
                .global_context
                .update_shadow_size_textures(&self.device, shadow_detail);
        }
    }

    pub fn get_backend_name(&self) -> String {
        self.adapter.get_info().backend.to_string()
    }

    pub fn get_present_mode_info(&self) -> PresentModeInfo {
        self.surface.as_ref().unwrap().present_mode_info()
    }

    pub fn get_window_size(&self) -> Vector2<usize> {
        self.surface.as_ref().unwrap().window_size()
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    pub fn wait_for_next_frame(&mut self) -> SurfaceTexture {
        // Before we wait for the next frame, we verify that the surface is still valid.
        if let Some(surface) = self.surface.as_mut()
            && surface.is_invalid()
        {
            #[cfg(feature = "debug")]
            profile_block!("re-configure surface and textures");

            surface.reconfigure();

            if let Some(engine_context) = self.engine_context.as_mut() {
                engine_context
                    .global_context
                    .update_screen_size_textures(&self.device, surface.window_screen_size());
            }
        }

        if self.limit_framerate {
            self.frame_pacer.wait_for_frame();
        }
        self.frame_pacer.begin_frame_stage(self.cpu_stage, Instant::now());

        self.surface.as_mut().expect("surface not set").acquire()
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    pub fn render_next_frame(&mut self, frame: SurfaceTexture, instruction: &RenderInstruction) {
        assert!(instruction.point_light_shadow_caster.len() <= NUMBER_OF_POINT_LIGHTS_WITH_SHADOWS);

        // Reclaim all staging buffers that the GPU has finished reading from.
        self.staging_belt.recall();

        // Calculate and stage the uploading of GPU data that is needed for the frame.
        let prepare_command_buffer = self.prepare_frame_data(instruction);

        // Record all draw commands.
        let (
            interface_command_buffer,
            picker_command_buffer,
            directional_shadow_command_buffer,
            point_shadow_command_buffer,
            geometry_command_buffer,
            screen_command_buffer,
        ) = self.draw_frame(&frame, instruction);

        // Queue all staging belt writes.
        self.staging_belt.finish();

        self.queue_picker_value();
        self.wait_and_submit_frame(
            prepare_command_buffer,
            interface_command_buffer,
            picker_command_buffer,
            directional_shadow_command_buffer,
            point_shadow_command_buffer,
            geometry_command_buffer,
            screen_command_buffer,
        );

        // Schedule the presentation of the frame.
        frame.present();

        self.frame_pacer.end_frame_stage(self.cpu_stage, Instant::now());
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    fn wait_and_submit_frame(
        &mut self,
        prepare_command_buffer: CommandBuffer,
        interface_command_buffer: CommandBuffer,
        picker_command_buffer: CommandBuffer,
        directional_shadow_command_buffer: CommandBuffer,
        point_shadow_command_buffer: CommandBuffer,
        geometry_command_buffer: CommandBuffer,
        screen_command_buffer: CommandBuffer,
    ) {
        // We have gathered all data for the next frame and can now wait until the GPU
        // is ready to accept the command buffers for the next frame. This is the
        // best time to resolve async operations like reading the piker value that need
        // to be synced with the GPU.
        self.device.poll(Maintain::Wait);
        self.queue.submit([
            prepare_command_buffer,
            interface_command_buffer,
            picker_command_buffer,
            directional_shadow_command_buffer,
            point_shadow_command_buffer,
            geometry_command_buffer,
            screen_command_buffer,
        ]);
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    fn queue_picker_value(&mut self) {
        if let Some(engine_context) = self.engine_context.as_ref() {
            engine_context
                .global_context
                .picker_value_buffer
                .queue_read_u64(self.picker_value.clone());
        }
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    fn prepare_frame_data(&mut self, instruction: &RenderInstruction) -> CommandBuffer {
        let context = self.engine_context.as_mut().unwrap();

        // We spawn a task for all the potentially long-running prepare functions.
        rayon::in_place_scope(|scope| {
            scope.spawn(|_| {
                context.directional_shadow_entity_drawer.prepare(&self.device, instruction);
                context.directional_shadow_model_drawer.prepare(&self.device, instruction);
            });
            scope.spawn(|_| {
                context.geometry_entity_drawer.prepare(&self.device, instruction);
                context.geometry_model_drawer.prepare(&self.device, instruction);
            });
            scope.spawn(|_| {
                context.interface_rectangle_drawer.prepare(&self.device, instruction);
            });
            scope.spawn(|_| {
                context.point_shadow_entity_drawer.prepare(&self.device, instruction);
                context.point_shadow_model_drawer.prepare(&self.device, instruction);
            });
            scope.spawn(|_| {
                context.screen_directional_light_drawer.prepare(&self.device, instruction);
                context.screen_effect_drawer.prepare(&self.device, instruction);
            });
            scope.spawn(|_| {
                context.screen_point_light_drawer.prepare(&self.device, instruction);
                context.screen_rectangle_drawer.prepare(&self.device, instruction);
            });
            #[cfg(feature = "debug")]
            scope.spawn(|_| {
                context.picker_marker_drawer.prepare(&self.device, instruction);
                context.screen_aabb_drawer.prepare(&self.device, instruction);
            });
            #[cfg(feature = "debug")]
            scope.spawn(|_| {
                context.screen_buffer_drawer.prepare(&self.device, instruction);
                context.screen_circle_drawer.prepare(&self.device, instruction);
            });

            context.global_context.prepare(&self.device, instruction);
            context.directional_shadow_pass_context.prepare(&self.device, instruction);
            context.point_shadow_pass_context.prepare(&self.device, instruction);
            context.picker_entity_drawer.prepare(&self.device, instruction);
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());

        let mut visitor = UploadVisitor {
            device: &self.device,
            staging_belt: &mut self.staging_belt,
            encoder: &mut encoder,
        };

        visitor.upload(&mut context.directional_shadow_entity_drawer);
        visitor.upload(&mut context.directional_shadow_model_drawer);
        visitor.upload(&mut context.directional_shadow_pass_context);
        visitor.upload(&mut context.geometry_entity_drawer);
        visitor.upload(&mut context.geometry_model_drawer);
        visitor.upload(&mut context.global_context);
        visitor.upload(&mut context.interface_rectangle_drawer);
        visitor.upload(&mut context.picker_entity_drawer);
        visitor.upload(&mut context.point_shadow_entity_drawer);
        visitor.upload(&mut context.point_shadow_model_drawer);
        visitor.upload(&mut context.point_shadow_pass_context);
        visitor.upload(&mut context.screen_directional_light_drawer);
        visitor.upload(&mut context.screen_effect_drawer);
        visitor.upload(&mut context.screen_point_light_drawer);
        visitor.upload(&mut context.screen_rectangle_drawer);

        #[cfg(feature = "debug")]
        {
            visitor.upload(&mut context.picker_marker_drawer);
            visitor.upload(&mut context.screen_aabb_drawer);
            visitor.upload(&mut context.screen_buffer_drawer);
            visitor.upload(&mut context.screen_circle_drawer);
        }

        encoder.finish()
    }

    #[cfg_attr(feature = "debug", korangar_debug::profile)]
    fn draw_frame(
        &mut self,
        frame: &SurfaceTexture,
        instruction: &RenderInstruction,
    ) -> (
        CommandBuffer,
        CommandBuffer,
        CommandBuffer,
        CommandBuffer,
        CommandBuffer,
        CommandBuffer,
    ) {
        let frame_view = &frame.texture.create_view(&TextureViewDescriptor::default());
        let engine_context = self.engine_context.as_mut().unwrap();
        #[cfg(feature = "debug")]
        let render_settings = &instruction.render_settings;

        let mut picker_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut interface_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut directional_shadow_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut point_shadow_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut geometry_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        let mut screen_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());

        rayon::in_place_scope(|scope| {
            // Picker Pass
            scope.spawn(|_| {
                let mut render_pass = engine_context.picker_render_pass_context.create_pass(
                    frame_view,
                    &mut picker_encoder,
                    &engine_context.global_context,
                    None,
                );

                engine_context
                    .picker_tile_drawer
                    .draw(&mut render_pass, instruction.map_picker_tile_vertex_buffer);
                engine_context.picker_entity_drawer.draw(&mut render_pass, instruction.entities);
                #[cfg(feature = "debug")]
                {
                    engine_context.picker_marker_drawer.draw(&mut render_pass, None);
                }

                drop(render_pass);

                // Copy the picker value from the texture into the buffer.
                let bytes_per_row = engine_context.global_context.picker_buffer_texture.get_bytes_per_row();
                let unpadded_texture_size = engine_context.global_context.picker_buffer_texture.get_unpadded_size();
                let x = (unpadded_texture_size.width - 1).min(instruction.picker_position.left as u32);
                let y = (unpadded_texture_size.height - 1).min(instruction.picker_position.top as u32);

                picker_encoder.copy_texture_to_buffer(
                    ImageCopyTexture {
                        texture: engine_context.global_context.picker_buffer_texture.get_texture(),
                        mip_level: 0,
                        origin: Origin3d { x, y, z: 0 },
                        aspect: TextureAspect::All,
                    },
                    ImageCopyBuffer {
                        buffer: engine_context.global_context.picker_value_buffer.get_buffer(),
                        layout: ImageDataLayout {
                            offset: 0,
                            bytes_per_row,
                            rows_per_image: None,
                        },
                    },
                    Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            });

            // Interface Pass
            scope.spawn(|_| {
                let mut render_pass = engine_context.interface_render_pass_context.create_pass(
                    frame_view,
                    &mut interface_encoder,
                    &engine_context.global_context,
                    instruction.clear_interface,
                );

                engine_context
                    .interface_rectangle_drawer
                    .draw(&mut render_pass, instruction.interface);
            });

            // Directional Shadow Caster Pass
            scope.spawn(|_| {
                let mut render_pass = engine_context.directional_shadow_pass_context.create_pass(
                    frame_view,
                    &mut directional_shadow_encoder,
                    &engine_context.global_context,
                    None,
                );

                let draw_data = ModelBatchDrawData {
                    batches: instruction.directional_model_batches,
                    instructions: instruction.directional_shadow_models,
                    #[cfg(feature = "debug")]
                    show_wireframe: false,
                };

                engine_context.directional_shadow_model_drawer.draw(&mut render_pass, draw_data);
                engine_context
                    .directional_shadow_indicator_drawer
                    .draw(&mut render_pass, instruction.indicator.as_ref());
                engine_context
                    .directional_shadow_entity_drawer
                    .draw(&mut render_pass, instruction.directional_shadow_entities);
            });

            // Point Shadow Caster Pass
            scope.spawn(|_| {
                (0..instruction.point_light_shadow_caster.len()).for_each(|shadow_caster_index| {
                    (0..6).for_each(|face_index| {
                        let pass_data = PointShadowData {
                            shadow_caster_index,
                            face_index,
                        };
                        let model_data = PointShadowModelBatchData {
                            pass_data,
                            caster: instruction.point_light_shadow_caster,
                            instructions: instruction.point_shadow_models,
                        };
                        let entity_data = PointShadowEntityBatchData {
                            pass_data,
                            caster: instruction.point_light_shadow_caster,
                            instructions: instruction.point_shadow_entities,
                        };

                        let mut render_pass = engine_context.point_shadow_pass_context.create_pass(
                            frame_view,
                            &mut point_shadow_encoder,
                            &engine_context.global_context,
                            pass_data,
                        );

                        engine_context.point_shadow_model_drawer.draw(&mut render_pass, &model_data);
                        engine_context.point_shadow_entity_drawer.draw(&mut render_pass, &entity_data);
                        engine_context
                            .point_shadow_indicator_drawer
                            .draw(&mut render_pass, instruction.indicator.as_ref());
                    });
                });
            });

            // Geometry Pass
            scope.spawn(|_| {
                let mut render_pass = engine_context.geometry_pass_context.create_pass(
                    frame_view,
                    &mut geometry_encoder,
                    &engine_context.global_context,
                    None,
                );

                let draw_data = ModelBatchDrawData {
                    batches: instruction.model_batches,
                    instructions: instruction.models,
                    #[cfg(feature = "debug")]
                    show_wireframe: instruction.render_settings.show_wireframe,
                };

                engine_context.geometry_model_drawer.draw(&mut render_pass, draw_data);
                engine_context
                    .geometry_indicator_drawer
                    .draw(&mut render_pass, instruction.indicator.as_ref());
                engine_context.geometry_entity_drawer.draw(&mut render_pass, instruction.entities);

                if let Some(map_water_vertex_buffer) = instruction.map_water_vertex_buffer.as_ref() {
                    engine_context.geometry_water_drawer.draw(&mut render_pass, map_water_vertex_buffer);
                }
            });

            // Screen Pass
            let mut render_pass =
                engine_context
                    .screen_pass_context
                    .create_pass(frame_view, &mut screen_encoder, &engine_context.global_context, None);

            #[cfg_attr(feature = "debug", korangar_debug::debug_condition(render_settings.show_ambient_light))]
            engine_context.screen_ambient_light_drawer.draw(&mut render_pass, None);
            #[cfg_attr(feature = "debug", korangar_debug::debug_condition(render_settings.show_directional_light))]
            engine_context.screen_directional_light_drawer.draw(&mut render_pass, None);
            #[cfg_attr(feature = "debug", korangar_debug::debug_condition(render_settings.show_point_lights))]
            engine_context.screen_point_light_drawer.draw(&mut render_pass, None);
            engine_context.screen_water_light_drawer.draw(&mut render_pass, None);

            #[cfg(feature = "debug")]
            {
                engine_context.screen_aabb_drawer.draw(&mut render_pass, None);
                engine_context.screen_circle_drawer.draw(&mut render_pass, None);
            }

            let rectangle_data = ScreenRectangleDrawInstruction {
                layer: Layer::Bottom,
                instructions: instruction.bottom_layer_rectangles,
            };

            engine_context.screen_rectangle_drawer.draw(&mut render_pass, rectangle_data);
            engine_context.screen_effect_drawer.draw(&mut render_pass, instruction.effects);

            #[cfg(feature = "debug")]
            {
                engine_context
                    .screen_buffer_drawer
                    .draw(&mut render_pass, &instruction.render_settings);
            }

            let rectangle_data = ScreenRectangleDrawInstruction {
                layer: Layer::Middle,
                instructions: instruction.middle_layer_rectangles,
            };

            engine_context.screen_rectangle_drawer.draw(&mut render_pass, rectangle_data);

            #[cfg(feature = "debug")]
            {
                #[cfg(feature = "debug")]
                engine_context.screen_aabb_drawer.draw(&mut render_pass, None);
            }

            if instruction.show_interface {
                engine_context.screen_overlay_drawer.draw(&mut render_pass, None);
            }

            let rectangle_data = ScreenRectangleDrawInstruction {
                layer: Layer::Top,
                instructions: instruction.top_layer_rectangles,
            };

            engine_context.screen_rectangle_drawer.draw(&mut render_pass, rectangle_data);
        });

        (
            picker_encoder.finish(),
            interface_encoder.finish(),
            directional_shadow_encoder.finish(),
            point_shadow_encoder.finish(),
            geometry_encoder.finish(),
            screen_encoder.finish(),
        )
    }
}

struct UploadVisitor<'a> {
    device: &'a Device,
    staging_belt: &'a mut StagingBelt,
    encoder: &'a mut CommandEncoder,
}

impl<'a> UploadVisitor<'a> {
    fn upload(&mut self, context: &mut impl Prepare) {
        context.upload(self.device, self.staging_belt, self.encoder);
    }
}