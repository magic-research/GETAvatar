# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d


# ----------------------------------------------------------------------------
class Loss:
    def accumulate_gradients(
            self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------
# Regulrarization loss for dmtet
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class StyleGAN2Loss(Loss):
    def __init__(
            self, device, G, D, r1_gamma=10, style_mixing_prob=0, pl_weight=0, eik_weight=1e-3,
            gamma_mask=10, blur_init_sigma=0, blur_fade_kimg=0, 
            blur_rgb_image=False, blur_normal_image=False):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.r1_gamma = r1_gamma
        self.min_r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.min_gamma_mask = gamma_mask
        self.gamma_mask = gamma_mask
        self.c_cam_dim = 16
        self.eik_weight = eik_weight

        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.blur_rgb_image = blur_rgb_image
        self.blur_normal_image = blur_normal_image

    def run_G(
            self, z, c, update_emas=False, return_shape=False, return_eikonal=False
    ):
        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])
        # Step 1: Map the sampled z code to w-space
        ws = self.G.mapping(z, c_cam_perm, update_emas=update_emas)
        geo_z = torch.randn_like(z)
        ws_geo = self.G.mapping_geo(
            geo_z, c_cam_perm,
            update_emas=update_emas)

        # Step 2: Apply style mixing to the latent code
        # NOTE: set to zero like avatargen
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws.shape[1]))
                # NOTE: c_cam_perm or c[:, :self.c_cam_dim]
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c_cam_perm, update_emas=False)[:, cutoff:]

                cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws_geo.shape[1]))
                ws_geo[:, cutoff:] = self.G.mapping_geo(torch.randn_like(z), c_cam_perm, update_emas=False)[:, cutoff:]

        # Step 3: Generate rendered image of 3D generated shapes.
        if return_shape:
            img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.G.synthesis(
                ws, c,
                ws_geo=ws_geo,
                return_shape=return_shape,
                return_eikonal=return_eikonal
            )
            return img, sdf, ws, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, ws_geo, sdf_reg_loss, eikonal_loss, render_return_value, part_img
        else:
            img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.G.synthesis(
                ws, c, 
                ws_geo=ws_geo,
                return_shape=return_shape,
                return_eikonal=return_eikonal
            )

        return img, ws, mask_pyramid, render_return_value, part_img

    def run_G_normal(
            self, z, c, update_emas=False, return_shape=False, return_eikonal=False
    ):
        # without gen_pose_cond
        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])
        # Step 1: Map the sampled z code to w-space
        ws = self.G.mapping(z, c_cam_perm, update_emas=update_emas)
        geo_z = torch.randn_like(z)
        ws_geo = self.G.mapping_geo(
            geo_z, c_cam_perm,
            update_emas=update_emas)

        # Step 2: Apply style mixing to the latent code
        # NOTE: set to zero like avatargen
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws.shape[1]))
                # ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
                # NOTE: c_cam_perm or c[:, :self.c_cam_dim]
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c_cam_perm, update_emas=False)[:, cutoff:]

                cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws_geo.shape[1]))
                ws_geo[:, cutoff:] = self.G.mapping_geo(torch.randn_like(z), c_cam_perm, update_emas=False)[:, cutoff:]

        # Step 3: Generate rendered image of 3D generated shapes.
        if return_shape:
            normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.G.synthesis.forward_normal(
                ws, c,
                ws_geo=ws_geo,
                return_shape=return_shape,
                return_eikonal=return_eikonal
            )
            return normal_img, sdf, ws, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, ws_geo, sdf_reg_loss, eikonal_loss, render_return_value, part_img
        else:
            normal_img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.G.synthesis.forward_normal(
                ws, c, 
                ws_geo=ws_geo,
                return_shape=return_shape,
                return_eikonal=return_eikonal)
           
        # return img, ws, syn_camera, mask_pyramid, render_return_value
        return normal_img, ws, mask_pyramid, render_return_value, part_img

    def run_D(self, img, c, blur_sigma=0, update_emas=False, mask_pyramid=None, part_img=None):
        if self.blur_rgb_image:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                with torch.autograd.profiler.record_function('blur'):
                    f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                    # img = upfirdn2d.filter2d(img, f / f.sum())
                    rgb_img, mask = img[:, :3], img[:, 3:]
                    rgb_img = upfirdn2d.filter2d(rgb_img, f / f.sum())
                    img = torch.cat((rgb_img, mask), dim=1)

        # if part_img is not None:
        if False:
            logits = self.D(img, c, update_emas=update_emas, mask_pyramid=mask_pyramid, part_img=part_img)
        else:
            logits = self.D(img, c, update_emas=update_emas, mask_pyramid=mask_pyramid)
        return logits
    
    def run_D_normal(self, img, c, blur_sigma=0, update_emas=False, mask_pyramid=None, part_img=None):
        if self.blur_normal_image:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                with torch.autograd.profiler.record_function('blur'):
                    f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                    # img = upfirdn2d.filter2d(img, f / f.sum())
                    normal_img, mask = img[:, :3], img[:, 3:]
                    normal_img = upfirdn2d.filter2d(normal_img, f / f.sum())
                    img = torch.cat((normal_img, mask), dim=1)
        # if part_img is not None:
        if False:
            logits = self.D.forward_normal(img, c, update_emas=update_emas, mask_pyramid=mask_pyramid, part_img=part_img)
        else:
            logits = self.D.forward_normal(img, c, update_emas=update_emas, mask_pyramid=mask_pyramid)
        return logits

    def accumulate_gradients(
            self, phase, real_img, real_norm_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Gnorm', 'Greg', 'Dmain', 'Dreg', 'Dnorm', 'Dreg_norm']
        if self.pl_weight == 0: # pl_weight == 0 by default
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0: # r1_gamma!=0 by default
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)

        if cur_nimg < 200 * 1e3:       # 200
            self.r1_gamma = 40.0
            self.gamma_mask = 40.0
        elif cur_nimg < 400 * 1e3:     # 400
            self.r1_gamma = 20.0
            self.gamma_mask = 20.0
        else:
            self.r1_gamma = 10.0
            self.gamma_mask = 10.0

        if self.r1_gamma < self.min_r1_gamma:
            self.r1_gamma = self.min_r1_gamma
        
        if self.gamma_mask < self.min_gamma_mask:
            self.gamma_mask = self.min_gamma_mask
        
        # FROM EG3D
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # First generate the rendered image of generated 3D shapes
                gen_img, gen_sdf, _gen_ws, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _gen_ws_geo, \
                sdf_reg_loss, eikonal_loss, render_return_value, gen_part_img = self.run_G(
                    gen_z, gen_c, return_shape=True
                )

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, mask_pyramid=mask_pyramid, part_img=gen_part_img)
                gen_logits, gen_logits_mask = gen_logits

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean() # mean here? seem ok
                training_stats.report('Loss/G/loss_rgb', loss_Gmain)

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Gmask = torch.nn.functional.softplus(-gen_logits_mask).mean()
                training_stats.report('Loss/G/loss_mask', loss_Gmask)
                loss_Gmain += loss_Gmask
                training_stats.report('Loss/G/loss', loss_Gmain)

                # Regularization loss for sdf prediction
                sdf_reg_loss_entropy = sdf_reg_loss_batch(gen_sdf, self.G.synthesis.dmtet_geometry.all_edges).mean() * 0.01
                training_stats.report('Loss/G/sdf_reg', sdf_reg_loss_entropy)
                loss_Gmain += sdf_reg_loss_entropy
                training_stats.report('Loss/G/sdf_reg_abs', sdf_reg_loss)
                loss_Gmain += sdf_reg_loss.mean()

                # eikonal loss
                if eikonal_loss is not None:
                    training_stats.report('Loss/G/eik_loss', eikonal_loss)
                    loss_Gmain += eikonal_loss * self.eik_weight

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gnorm: Maximize logits for generated images.
        if phase in ['Gnorm']:
            with torch.autograd.profiler.record_function('Gnorm_forward'):
                # First generate the rendered image of generated 3D shapes
                # remove gen_camera
                gen_normal_img, gen_sdf, _gen_ws, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _gen_ws_geo, \
                sdf_reg_loss, eikonal_loss, render_return_value, gen_part_normal_img = self.run_G_normal(
                    gen_z, gen_c, return_shape=True, return_eikonal=True
                )

                gen_logits = self.run_D_normal(gen_normal_img, gen_c, blur_sigma=blur_sigma, mask_pyramid=mask_pyramid, part_img=gen_part_normal_img)
                gen_logits_normal, gen_logits_mask = gen_logits

                if real_norm_img is not None:
                    training_stats.report('Loss/scores/fakes_normal', gen_logits_normal)
                    training_stats.report('Loss/signs/fakes_normal', gen_logits_normal.sign())
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits_normal).mean() # mean here? seem ok
                    training_stats.report('Loss/G/loss_normal', loss_Gmain)
                else:
                    loss_Gmain = 0.0

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Gmask = torch.nn.functional.softplus(-gen_logits_mask).mean()
                training_stats.report('Loss/G/loss_mask', loss_Gmask)
                loss_Gmain += loss_Gmask
                training_stats.report('Loss/G/loss_normal_total', loss_Gmain)

                # Regularization loss for sdf prediction
                sdf_reg_loss_entropy = sdf_reg_loss_batch(gen_sdf, self.G.synthesis.dmtet_geometry.all_edges).mean() * 0.01
                training_stats.report('Loss/G/sdf_reg', sdf_reg_loss_entropy)
                loss_Gmain += sdf_reg_loss_entropy
                training_stats.report('Loss/G/sdf_reg_abs', sdf_reg_loss)
                loss_Gmain += sdf_reg_loss.mean()

                # eikonal loss
                if eikonal_loss is not None:
                    training_stats.report('Loss/G/eik_loss', eikonal_loss)
                    loss_Gmain += eikonal_loss * self.eik_weight

            with torch.autograd.profiler.record_function('Gnorm_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # We didn't have Gpl regularization

        #######################################################
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # First generate the rendered image of generated 3D shapes
                gen_img, _gen_ws, mask_pyramid, render_return_value, gen_part_img = self.run_G(
                    gen_z, gen_c, update_emas=True)

                gen_logits = self.run_D(
                    gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, mask_pyramid=mask_pyramid, part_img=gen_part_img)

                gen_logits, gen_logits_mask = gen_logits

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_gen_rgb', loss_Dgen)

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Dgen_mask = torch.nn.functional.softplus(
                    gen_logits_mask).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_gen_mask', loss_Dgen_mask)
                loss_Dgen += loss_Dgen_mask

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        #######################################################
        # Dnorm: Minimize logits for generated normal images.
        loss_Dgen_normal = 0
        if phase in ['Dnorm']:
            with torch.autograd.profiler.record_function('Dgen_norm_forward'):
                # First generate the rendered image of generated 3D shapes
                self.G.requires_grad_(True)
                gen_normal_img, _gen_ws, mask_pyramid, render_return_value, gen_part_normal_img = self.run_G_normal(
                    gen_z, gen_c, update_emas=True)
                self.G.requires_grad_(False)
                if gen_part_normal_img is not None:
                    gen_part_normal_img = gen_part_normal_img.detach()

                gen_logits = self.run_D_normal(
                    gen_normal_img.detach(), gen_c, blur_sigma=blur_sigma, update_emas=True, mask_pyramid=mask_pyramid, part_img=gen_part_normal_img)

                gen_logits_normal, gen_logits_mask = gen_logits

                if real_norm_img is not None:
                    training_stats.report('Loss/scores/fake_normal', gen_logits_normal)
                    training_stats.report('Loss/signs/fake_normal', gen_logits_normal.sign())
                    loss_Dgen_normal = torch.nn.functional.softplus(gen_logits_normal).mean()  # -log(1 - sigmoid(gen_logits))
                    training_stats.report('Loss/D/loss_gen_normal', loss_Dgen_normal)
                else:
                    loss_Dgen_normal = 0.0

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Dgen_mask = torch.nn.functional.softplus(
                    gen_logits_mask).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_gen_mask', loss_Dgen_mask)
                loss_Dgen_normal += loss_Dgen_mask

            with torch.autograd.profiler.record_function('Dgen_norm_backward'):
                loss_Dgen_normal.mean().mul(gain).backward()

        # Dnormal: Maximize logits for real normal images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dnorm', 'Dreg_norm'] and (real_norm_img is not None):
            name = 'Dreal_norm' if phase == 'Dnorm' else 'Dr1_norm' if phase == 'Dreg_norm' else 'Dreal_norm_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Optimize for the real image
                real_norm_img_tmp = real_norm_img.detach().requires_grad_(phase in ['Dreg_norm'])
                if self.G.synthesis.part_disc:
                    canonical_mapping_kwargs = self.G.synthesis.get_canonical_mapping_quick(real_c)
                    real_norm_part_img = self.G.synthesis.get_part(real_norm_img_tmp, real_c, canonical_mapping_kwargs).detach().requires_grad_(phase in ['Dreg_norm'])
                else:
                    real_norm_part_img = None

                real_logits = self.run_D_normal(real_norm_img_tmp, real_c, blur_sigma=blur_sigma, part_img=real_norm_part_img)
                real_logits_normal, real_logits_mask = real_logits

                training_stats.report('Loss/scores/real_normal', real_logits_normal)
                training_stats.report('Loss/signs/real_normal', real_logits_normal.sign())

                training_stats.report('Loss/scores/real_mask', real_logits_mask)
                training_stats.report('Loss/signs/real_mask', real_logits_mask.sign())

                loss_Dreal_normal = 0
                if phase in ['Dnorm']:
                    loss_Dreal_normal = torch.nn.functional.softplus(-real_logits_normal).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_normal', loss_Dreal_normal)

                    loss_Dreal_mask = torch.nn.functional.softplus(
                        -real_logits_mask).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_mask', loss_Dreal_mask)
                    loss_Dreal_normal += loss_Dreal_mask
                    training_stats.report('Loss/D/loss_total', loss_Dgen_normal + loss_Dreal_normal)

                loss_Dr1_normal = 0
                # Compute R1 regularization for discriminator
                if phase in ['Dreg_norm']:
                    # Compute R1 regularization for discriminator of normal image
                    with torch.autograd.profiler.record_function('r1_grads_norm'), conv2d_gradfix.no_weight_gradients():
                        r1_grads_normal = torch.autograd.grad(
                            outputs=[real_logits_normal.sum()], inputs=[real_norm_img_tmp], create_graph=True, only_inputs=True)[0]

                    r1_penalty_normal = r1_grads_normal.square().sum([1, 2, 3])
                    loss_Dr1_normal = r1_penalty_normal.mean() * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_normal', r1_penalty_normal)
                    training_stats.report('Loss/D/reg_normal', loss_Dr1_normal)
                    # Compute R1 regularization for discriminator of Mask image
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads_mask = \
                            torch.autograd.grad(
                                outputs=[real_logits_mask.sum()], inputs=[real_norm_img_tmp], create_graph=True,
                                only_inputs=True)[0]

                    r1_penalty_mask = r1_grads_mask.square().sum([1, 2, 3])
                    loss_Dr1_mask = r1_penalty_mask.mean() * (self.gamma_mask / 2)
                    training_stats.report('Loss/r1_penalty_mask', r1_penalty_mask)
                    training_stats.report('Loss/D/reg_mask', loss_Dr1_mask)
                    loss_Dr1_normal += loss_Dr1_mask
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal_normal + loss_Dr1_normal).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Optimize for the real image
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg'])
                if self.G.synthesis.part_disc:
                    canonical_mapping_kwargs = self.G.synthesis.get_canonical_mapping_quick(real_c)
                    real_part_img = self.G.synthesis.get_part(real_img_tmp, real_c, canonical_mapping_kwargs).detach().requires_grad_(phase in ['Dreg'])
                else:
                    real_part_img = None

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, part_img=real_part_img)
                real_logits, real_logits_mask = real_logits

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                training_stats.report('Loss/scores/real_mask', real_logits_mask)
                training_stats.report('Loss/signs/real_mask', real_logits_mask.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_rgb', loss_Dreal)

                    loss_Dreal_mask = torch.nn.functional.softplus(
                        -real_logits_mask).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_mask', loss_Dreal_mask)
                    loss_Dreal += loss_Dreal_mask
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                # Compute R1 regularization for discriminator
                if phase in ['Dreg']:
                    # Compute R1 regularization for discriminator of RGB image
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]

                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty.mean() * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
                    # Compute R1 regularization for discriminator of Mask image
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads_mask = \
                            torch.autograd.grad(
                                outputs=[real_logits_mask.sum()], inputs=[real_img_tmp], create_graph=True,
                                only_inputs=True)[0]

                    r1_penalty_mask = r1_grads_mask.square().sum([1, 2, 3])
                    loss_Dr1_mask = r1_penalty_mask.mean() * (self.gamma_mask / 2)
                    training_stats.report('Loss/r1_penalty_mask', r1_penalty_mask)
                    training_stats.report('Loss/D/reg_mask', loss_Dr1_mask)
                    loss_Dr1 += loss_Dr1_mask
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()